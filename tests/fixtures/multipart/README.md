# Multipart form fixtures

Forms as four HTTP clients send them, for `multipart_test`: each file holds
the request's `Content-Type` line, an empty line and the body. Every client
sent the same fields: `model` and `prompt` as text (`Grüße "quoted"`), a WAV
`file`, and two `image[]` files; the file contents are short byte patterns
with CR, LF and `--` in them.

| File | Client |
| --- | --- |
| `curl.form` | curl 8.18.0, `-F` |
| `requests.form` | Python requests 2.32.5 (`files=`, `data=`) |
| `node.form` | Node.js 22.22.1 `fetch` with `FormData`, the encoding browsers use |
| `go.form` | Go 1.26.8 `mime/multipart` |

They were captured by a local listener that stored the `Content-Type` header
and the body of one POST from each client. The test needs no client to run.
