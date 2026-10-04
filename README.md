# Jev Context Selector

本项目是本地实验台：上传 PDF、抽取正文、切块，并对连续问题生成 Full、Jev 和明确未配置的 RAG 对照记录。SQLite 数据默认保存于 `.runtime/jev_context.sqlite3`。

## 运行

```powershell
python -m pip install -r requirements.txt
copy .env.example .env
python -m uvicorn app.main:app --reload
```

`.env` 只留在本机和服务器上，不要提交。仓库里只有空的 `.env.example`。

服务器上把代码拉到 `/home/ubuntu/jev`，复制 `.env.example` 为 `.env` 并填入密钥，再建虚拟环境安装依赖。应用只听本机 8000。同机的 runtime 已占用公网 8080，所以本项目对外用 `deploy/Caddyfile` 的 8081。进程用 `deploy/jev.service` 交给 systemd，退出登录后仍会运行。腾讯云防火墙和系统防火墙都要放行 TCP 8081。没有域名时，简历地址是 `http://公网IP:8081`。

## API

- `POST /workspaces`：上传 PDF，建立可复用工作区，并保存切块规则快照、计数口径和配置指纹。同一次对话里的 PDF 属于同一批。
- `POST /workspaces/{id}/documents`：只添加这一份 PDF 和它的块。超过上限时拒绝这次添加，不截断。
- `DELETE /workspaces/{id}/documents/{document_id}`：只删除这一份 PDF 和它的块。已经做出的回答保留当时的选文。
- `GET /workspaces/{id}`：读取当前文档、活动块集合、规则来源和问题历史。
- `POST /workspaces/{id}/questions`：JSON `{ "question": "...", "mode": "jev|full|rag|all" }`。
- `POST /workspaces/{id}/rechunk`：显式按当前配置建立新块集合；旧块和旧选文保留，旧选文变为不可继续作答。
- `GET /questions/{id}/runs`：读取选文、最新作答和 `answers` 全部历史、usage 来源与指标状态。
- `POST /questions/{question_id}/runs/{selection_id}/retry`：沿用同一有效块集合创建新的选文记录，保留旧运行。
- `POST /selections/{selection_id}/answers/retry`：只重试生成，不重新选文；拒绝未知、legacy、obsolete 或过期输入。
- `POST /selections/{selection_id}/obsolete`：将旧选文标为 `obsolete`。
- `POST /questions/{question_id}/comparisons`：创建只读独立比较快照，不调用模型。
- `GET /questions/{question_id}/comparisons`、`GET /comparisons/{id}`：读取比较历史及当前引用有效性。
- `GET/POST /workspaces/{id}/ask`、`GET /questions/{id}/history`：同工作区继续提问和查看完整运行/作答历史。

比较记录区分正文近似规模、完整块上下文近似规模、选中上下文、相同口径节省率、reported/estimated/unavailable usage、阶段耗时和费用状态。Full 选择费用为 0；未计算的生成费用和总费用保持 `null/not_calculated`。普通提问没有标准答案，Answer-F1 和 Evidence-F1 保持 `not_run`。题库题带上参考答案和证据位置后，全文、Jev、标准检索分别计分；没有完成的作答标成 `unanswered`，不补 0。

`rag` 当前明确返回未配置状态；Full 不调用 Jev。空选文、选文部分失败和作答失败不会伪造成功答案。错误文本会脱敏，不输出密钥。

## 验证

```powershell
python -m pytest tests -q
python -X utf8 tests/audit_acceptance.py
python -X utf8 tests/audit_frontend_followup.py
python -X utf8 tests/audit_lifecycle.py
python -X utf8 tests/audit_proof_boundaries.py
```

真实服务 smoke 只在已有密钥配置时运行，连通性与答案质量分开记录。后台返修仅做离线验证；前台已通过小规模真实链路验证，详见 design/14-最终验收与交接.md。不要打印密钥或 `.env` 内容。
