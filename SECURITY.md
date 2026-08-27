# Security

HistoraJ is a local (or self-hosted) Streamlit app. It **accepts file uploads** and an optional **pandas `query` string**. That is enough to treat a public multi-user deploy as untrusted input.

## What the app does with your data

- Uploaded CSV / Excel / Stata files are parsed in-process with pandas and kept in Streamlit `session_state` for that browser session.
- Files are not written to a project folder by this code, and there is no account store.
- Closing the session drops the in-memory table.

## Risks if you host it

1. **Uploads.** A crafted file can be large (default Streamlit limit is 200 MB) or can fail parsers in surprising ways. Excel macro sheets are not executed by `openpyxl` reads, but you should still only upload data you are willing to load into Python.
2. **Condition box.** The filter uses `DataFrame.query`. That is a small expression language, not a sandbox. On a shared server, a hostile expression is a code-execution concern. Run this app yourself, or put it behind authentication and a process sandbox if others can type in the box.
3. **No authentication.** Anyone who can open the URL can upload data into that process.

## Reporting a vulnerability

Please **do not** open a public issue for a vulnerability.

- Use [GitHub private vulnerability reporting](https://github.com/RajeshTharyan/HISTORAJ/security/advisories/new) if it is enabled on this repository, or
- Email the maintainer at **rajeshtharyan@gmail.com** with steps to reproduce.

This is a small MIT-licensed demo. There is no SLA; reports will be read and addressed as time allows.
