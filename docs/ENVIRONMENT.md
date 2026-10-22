# Environment

Captured in Phase 0 with `tinygpt env`.

```json
{
  "os": "Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.39",
  "python": "3.12.3",
  "torch": "2.14.1+cpu",
  "logical_cores": 22,
  "torch_threads": 11,
  "cuda": false,
  "mps": false,
  "ram_gb": 15.3
}
```

CPU: Intel Core Ultra 9 185H (hybrid: performance + efficiency cores), 22 logical threads.
Repo lives on the Linux filesystem (`~/code/tiny-gpt`), not `/mnt/c`, for IO speed.
