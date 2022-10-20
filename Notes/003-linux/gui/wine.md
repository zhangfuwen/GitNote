---

title: wine

---

# wine architecture

[winehq wiki: Architecture overview](https://wiki.winehq.org/Wine_Developer%27s_Guide/Architecture_Overview)

## windows 9x architecture

```
+---------------------+        \
|     Windows EXE     |         } application
+---------------------+        /

+---------+ +---------+        \
| Windows | | Windows |         \ application & system DLLs
|   DLL   | |   DLL   |         /
+---------+ +---------+        /

+---------+ +---------+      \
|  GDI32  | | USER32  |       \
|   DLL   | |   DLL   |        \
+---------+ +---------+         } core system DLLs
+---------------------+        /
|     Kernel32 DLL    |       /
+---------------------+      / 

+---------------------+        \
|     Win9x kernel    |         } kernel space
+---------------------+        /

+---------------------+       \
|  Windows low-level  |        \ drivers (kernel space)
|       drivers       |        /
+---------------------+       /
```

## windows NT architecture

```

+---------------------+                      \
|     Windows EXE     |                       } application
+---------------------+                      /

+---------+ +---------+                      \
| Windows | | Windows |                       \ application & system DLLs
|   DLL   | |   DLL   |                       /
+---------+ +---------+                      /

+---------+ +---------+     +-----------+   \
|  GDI32  | |  USER32 |     |           |    \
|   DLL   | |   DLL   |     |           |     \
+---------+ +---------+     |           |      \ core system DLLs
+---------------------+     |           |      / (on the left side)
|    Kernel32 DLL     |     | Subsystem |     /
|  (Win32 subsystem)  |     |Posix, OS/2|    /
+---------------------+     +-----------+   / 

+---------------------------------------+
|               NTDLL.DLL               |
+---------------------------------------+

+---------------------------------------+     \
|               NT kernel               |      } NT kernel (kernel space)
+---------------------------------------+     /
+---------------------------------------+     \
|       Windows low-level drivers       |      } drivers (kernel space)
+---------------------------------------+     /

```

## wine architecture

```
+---------------------+                                  \
|     Windows EXE     |                                   } application
+---------------------+                                  /

+---------+ +---------+                                  \
| Windows | | Windows |                                   \ application & system DLLs
|   DLL   | |   DLL   |                                   /
+---------+ +---------+                                  /

+---------+ +---------+     +-----------+  +--------+  \
|  GDI32  | |  USER32 |     |           |  |        |   \
|   DLL   | |   DLL   |     |           |  |  Wine  |    \
+---------+ +---------+     |           |  | Server |     \ core system DLLs
+---------------------+     |           |  |        |     / (on the left side)
|    Kernel32 DLL     |     | Subsystem |  | NT-like|    /
|  (Win32 subsystem)  |     |Posix, OS/2|  | Kernel |   /
+---------------------+     +-----------+  |        |  / 
                                           |        |
+---------------------------------------+  |        |
|                 NTDLL                 |  |        |
+---------------------------------------+  +--------+

+---------------------------------------+               \
|            Wine executable            |                } unix executable
+---------------------------------------+               /
+---------------------------------------------------+   \
|                   Wine drivers                    |    } Wine specific DLLs
+---------------------------------------------------+   /

+------------+    +------------+     +--------------+   \
|    libc    |    |   libX11   |     |  other libs  |    } unix shared libraries
+------------+    +------------+     +--------------+   /  (user space)

+---------------------------------------------------+   \
|         Unix kernel (Linux,*BSD,Solaris,OS/X)     |    } (Unix) kernel space
+---------------------------------------------------+   /
+---------------------------------------------------+   \
|                 Unix device drivers               |    } Unix drivers (kernel space)
+---------------------------------------------------+   /
```

Wine must at least completely replace the `Big Three` DLLs (`KERNEL/KERNEL32`, `GDI/GDI32`, and `USER/USER32`), which all other DLLs are layered on top of. But since Wine is (for various reasons) leaning towards the NT way of implementing things, the `NTDLL` is another core DLL to be implemented in Wine, and many KERNEL32 and ADVAPI32 features will be implemented through the NTDLL.