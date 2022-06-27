---

title: zenity

---


# wget显示进度

```bash
    wget --progress=bar:force "http://base.url.here/filename.txt" -O/your/destination/and/filename 2>&1 | zenity --title="File transfer in progress!" --progress --auto-close --auto-kill
```


# wget via ssh
```bash
ssh -C user@hostB "wget -O- http://website-C" >> file-from-website-C
```

# dd

```bash
dd if=.. of=.. status=progress oflag=direct,sync bs=10M
```