package api_tests

import (
	"encoding/json"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func TestLLMAnthropicMessagesWithoutV1(t *testing.T) {
	for _, path := range []string{"llm/v1/messages", "llm/messages"} {
		t.Run(path, func(t *testing.T) {
			resp, body := ExecRequest(t, R{
				Method: http.MethodPost,
				Path:   path,
				Body:   `{"max_tokens": 10, "messages": [{"role": "user", "content": "Hi"}]}`,
			})
			assert.Equal(t, http.StatusOK, resp.StatusCode)
			assert.Contains(t, body, `"type": "message"`)
		})
	}
}

func TestLLMChatCompletionsWithContentParts(t *testing.T) {
	s := assert.New(t)
	resp, body := ExecRequest(t, R{
		Method: http.MethodPost,
		Path:   "llm/v1/chat/completions",
		Body:   `{"messages": [{"role": "user", "content": [{"type": "text", "text": "Hi there"}]}]}`,
	})
	s.Equal(http.StatusOK, resp.StatusCode)
	s.Equal("application/json", resp.Header.Get("Content-Type"))

	var result struct {
		Usage struct {
			PromptTokens int `json:"prompt_tokens"`
		} `json:"usage"`
	}
	if s.NoError(json.Unmarshal([]byte(body), &result), "body: %s", body) {
		s.Equal(4, result.Usage.PromptTokens) // "user: Hi there\n"
	}
}

func TestLLMCompletionsWithCustomContent(t *testing.T) {
	s := assert.New(t)
	resp, body := ExecRequest(t, R{
		Method: http.MethodPost,
		Path:   "llm/v1/completions",
		Body:   `{"prompt": "hi", "httpbun": {"content": "custom"}}`,
	})
	s.Equal(http.StatusOK, resp.StatusCode)
	s.Equal("application/json", resp.Header.Get("Content-Type"))

	var result struct {
		Choices []struct {
			Text string `json:"text"`
		} `json:"choices"`
	}
	if s.NoError(json.Unmarshal([]byte(body), &result), "body: %s", body) && s.Len(result.Choices, 1) {
		s.Equal("custom", result.Choices[0].Text)
	}
}

func TestLLMLargeRequestBody(t *testing.T) {
	// The body size limit for LLM endpoints, `maxBodySize` in the llm package.
	const maxBodySize = 1 << 20

	message := func(size int) string {
		return `{"messages": [{"role": "user", "content": "` + strings.Repeat("a", size) + `"}]}`
	}

	// Larger than the 10KB limit for other endpoints.
	for _, path := range []string{"llm/v1/chat/completions", "llm/v1/messages"} {
		t.Run(path, func(t *testing.T) {
			resp, body := ExecRequest(t, R{Method: http.MethodPost, Path: path, Body: message(100_000)})
			assert.Equal(t, http.StatusOK, resp.StatusCode)
			assert.Equal(t, "application/json", resp.Header.Get("Content-Type"))
			assert.True(t, json.Valid([]byte(body)), "body: %s", body)
		})
	}

	t.Run("too large", func(t *testing.T) {
		resp, body := ExecRequest(t, R{
			Method: http.MethodPost,
			Path:   "llm/v1/chat/completions",
			Body:   message(maxBodySize),
		})
		assert.Equal(t, http.StatusRequestEntityTooLarge, resp.StatusCode)
		assert.Equal(t, "application/json", resp.Header.Get("Content-Type"))
		assert.Contains(t, body, `"type": "request_too_large"`)
	})
}

func TestLLMStreamLargeContent(t *testing.T) {
	// Streams are at most `maxStreamChunks` deltas in the llm package, 50ms apart, so even the largest content finishes
	// in about 30 seconds.
	const maxStreamChunks = 600
	const bound = 60 * time.Second

	content := strings.Repeat("word ", 200_000)
	contentJSON, _ := json.Marshal(content)

	for _, tc := range []struct {
		path string
		body string
		// delta returns the text in a streamed event, and whether the event is a delta.
		delta func(event map[string]any) (string, bool)
		last  string
	}{
		{
			path: "llm/v1/chat/completions",
			body: `{"stream": true, "messages": [], "httpbun": {"content": ` + string(contentJSON) + `}}`,
			delta: func(event map[string]any) (string, bool) {
				choice := event["choices"].([]any)[0].(map[string]any)
				text, ok := choice["delta"].(map[string]any)["content"].(string)
				return text, ok
			},
			last: "[DONE]",
		},
		{
			path: "llm/v1/completions",
			body: `{"stream": true, "prompt": "hi", "httpbun": {"content": ` + string(contentJSON) + `}}`,
			delta: func(event map[string]any) (string, bool) {
				choice := event["choices"].([]any)[0].(map[string]any)
				text := choice["text"].(string)
				return text, choice["finish_reason"] == nil
			},
			last: "[DONE]",
		},
		{
			path: "llm/v1/responses",
			body: `{"stream": true, "input": "hi", "httpbun": {"output_text": ` + string(contentJSON) + `}}`,
			delta: func(event map[string]any) (string, bool) {
				text, _ := event["delta"].(string)
				return text, event["type"] == "response.output_text.delta"
			},
			last: "response.completed",
		},
		{
			path: "llm/v1/messages",
			body: `{"stream": true, "messages": [], "httpbun": {"content": ` + string(contentJSON) + `}}`,
			delta: func(event map[string]any) (string, bool) {
				if event["type"] != "content_block_delta" {
					return "", false
				}
				return event["delta"].(map[string]any)["text"].(string), true
			},
			last: "message_stop",
		},
	} {
		t.Run(tc.path, func(t *testing.T) {
			t.Parallel()
			s := assert.New(t)

			start := time.Now()
			resp, body := ExecRequest(t, R{Method: http.MethodPost, Path: tc.path, Body: tc.body, Timeout: bound})
			s.Equal(http.StatusOK, resp.StatusCode)

			var text strings.Builder
			deltas := 0
			lastEvent := ""
			for _, line := range strings.Split(body, "\n") {
				data, ok := strings.CutPrefix(line, "data: ")
				if !ok {
					continue
				}
				lastEvent = data
				if data == "[DONE]" {
					continue
				}
				var event map[string]any
				if !s.NoError(json.Unmarshal([]byte(data), &event)) {
					return
				}
				if eventType, ok := event["type"].(string); ok {
					lastEvent = eventType
				}
				if delta, ok := tc.delta(event); ok {
					text.WriteString(delta)
					deltas++
				}
			}

			s.Less(time.Since(start), bound, "finished within the bound")
			s.Equal(tc.last, lastEvent, "stream finished")
			s.LessOrEqual(deltas, maxStreamChunks)
			s.Equal(content, text.String(), "whole content streamed")
		})
	}
}

func TestLLMStreamUsage(t *testing.T) {
	// With `stream_options.include_usage`, OpenAI-compatible streams send one extra chunk just before [DONE], with an
	// empty choices array and the usage for the whole request, and a null `usage` on every other chunk. See
	// https://platform.openai.com/docs/api-reference/chat/stream and
	// https://platform.openai.com/docs/api-reference/completions/stream.
	for _, tc := range []struct {
		path string
		body string
	}{
		{
			path: "llm/v1/chat/completions",
			body: `{"stream": true, "messages": [{"role": "user", "content": "Hello"}], "stream_options": {"include_usage": true}}`,
		},
		{
			path: "llm/v1/completions",
			body: `{"stream": true, "prompt": "Hello", "stream_options": {"include_usage": true}}`,
		},
	} {
		t.Run(tc.path, func(t *testing.T) {
			s := assert.New(t)
			resp, body := ExecRequest(t, R{Method: http.MethodPost, Path: tc.path, Body: tc.body})
			s.Equal(http.StatusOK, resp.StatusCode)

			var usage map[string]any
			for _, line := range strings.Split(body, "\n") {
				data, ok := strings.CutPrefix(line, "data: ")
				if !ok || data == "[DONE]" {
					continue
				}
				var event map[string]any
				if !s.NoError(json.Unmarshal([]byte(data), &event)) {
					return
				}
				s.Contains(event, "usage")
				if u, ok := event["usage"].(map[string]any); ok {
					usage = u
					s.Empty(event["choices"], "usage chunk has an empty choices array")
				}
			}

			if s.NotNil(usage, "stream ended without a usage chunk") {
				promptTokens := usage["prompt_tokens"].(float64)
				completionTokens := usage["completion_tokens"].(float64)
				s.Greater(promptTokens, float64(0))
				s.Greater(completionTokens, float64(0))
				s.Equal(promptTokens+completionTokens, usage["total_tokens"])
			}
		})
	}

	// Without include_usage, no chunk carries usage.
	for _, tc := range []struct{ path, body string }{
		{"llm/v1/chat/completions", `{"stream": true, "messages": [{"role": "user", "content": "Hello"}]}`},
		{"llm/v1/completions", `{"stream": true, "prompt": "Hello"}`},
	} {
		t.Run(tc.path+" without include_usage", func(t *testing.T) {
			resp, body := ExecRequest(t, R{Method: http.MethodPost, Path: tc.path, Body: tc.body})
			assert.Equal(t, http.StatusOK, resp.StatusCode)
			assert.NotContains(t, body, `"usage"`, "usage is only sent when explicitly asked for")
		})
	}
}
