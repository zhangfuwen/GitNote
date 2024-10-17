---

title: url scheme handler and mime type

---


# 什么是url scheme handler?

在操作系统中，不同的应用之间可以相互拉起，但又不知道自己拉起的应用到底是谁。

比如说 你想拉起邮件客户端，但又不确定系统里安装了哪个邮件客户端，比如说你可能有thunderbird, 也可能是outlook。如何能无差别地拉起一个邮件客户端？

你可以这样:

```bash
xdg-open mailto:zhangfuwen@foxmail.com
```

mailto:zhangfuwen@foxmail.com就是一个url scheme。

系统里有一个映射关系，这个映射关系可能通过任何方式维护，在不同的操作系统可能不一样。或者同一个操作系统中也存在不同的机制。

# Linux

## 通过desktop file声明

```desktop
[Desktop Entry]
Version=1.0
Type=Application
Name=dnfurl
Exec=dnfurl %U
Terminal=false
NoDisplay=true
MimeType=x-scheme-handler/dnf
```

https://blog.kagesenshi.org/2021/02/custom-url-scheme-handler.html

显示每次查一下所有的desktop文件是不太可能的，其实OS会维护一个cache，可以通过命令更新：

```bash
cd /usr/share/applications/
update-desktop-database
```

## 直接修改cache文件

所谓的cache文件可能就是下面的几个文件：

```
/usr/share/applications/defaults.list
~/.local/share/applications/mimeapps.list
~/.config/mimeapps.list
```

我们修改文件可以直接达到效果：

```bash

sudo su

echo "x-scheme-handler/ddg=org.gnome.Totem.desktop" >> /usr/share/applications/defaults.list

xdg-open ddg:asdfadfl
```

以上代码可以打开totem，当然了，totem并不认识ddg：asdfadf1是什么，所以它会报错。

## 应用如何响应呢？

如果我们自己写一个应用，建立了映射关系，我们如何收到这个请求呢？

其实它就是做为命令行参数传给我们的应用的。

我们看看totem的desktop file:

```
➜  Code head /usr/share/applications/org.gnome.Totem.desktop
[Desktop Entry]
Name=Videos
Comment=Play movies
Exec=totem %U
Icon=org.gnome.Totem
DBusActivatable=true
````

可以看到Exec字段跟的是`totem %U`，这里的'%U'就是url list的意思，这个可以参考：https://specifications.freedesktop.org/desktop-entry-spec/desktop-entry-spec-latest.html#exec-variables

# mime type

与url scheme handler 紧密相关的是mime type。

mime type的处理有一个通用的脚本，叫xdg-mime，它实际上是先检测一下DE，如果是kde就调用`kbuildsycoca`这个软件，如果是gnome就调用update-mime-database这个软件。

xdg-mine这个脚本是`xdg-utils`包的一部分，这个包里面只是几个脚本，可以通过`apt source xdg-utils`下载代码看一下，也可以在[github上直接看](https://github.com/freedesktop/xdg-utils/tree/master/scripts)。

## update-mime-database

update-mime-database是一个可执行文件，它存在于包`shared-mime-info`里，可以通过`apt source shared-mime-info`查看代码，也可以[在github上直接查看](https://github.com/freedesktop/xdg-shared-mime-info/tree/master/src)。

它接受一个目录做为参数，这个目录就是存放mime定义文件的地方，这个目录包括：

```
~/.local/share/mime/
/usr/share/mime

```

里面通过一个xml文件描述了一些mime类型，主要是它的名字和识别方法，识别方法包括通过后缀名识别和通过文件头部识别两种方法。
举个例子：

 cat ~/.local/share/mime/packages/view3dscene.xml

```
<?xml version="1.0" encoding="UTF-8"?>
<mime-info xmlns="http://www.freedesktop.org/standards/shared-mime-info">

  <!-- VRML, 3DS mime spec is copied from
       /usr/share/mime/packages/freedesktop.org.xml in Debian.

       I also added Gzip compressed extensions to this, which is
       not 100% correct solution, but see below for comments about
       this for X3D. -->

  <mime-type type="model/vrml">
    <sub-class-of type="text/plain"/>
    <comment>VRML document</comment>
    <comment xml:lang="az">VRML sənədi</comment>
    <comment xml:lang="bg">Документ — VRML</comment>
    <comment xml:lang="ca">document VRML</comment>
    <comment xml:lang="cs">Dokument VRML</comment>
    <comment xml:lang="cy">Dogfen VRML</comment>
    <comment xml:lang="da">VRML-dokument</comment>
    <comment xml:lang="de">VRML-Dokument</comment>
    <comment xml:lang="el">έγγραφο VRML</comment>
    <comment xml:lang="en_GB">VRML document</comment>
    <comment xml:lang="eo">VRML-dokumento</comment>
    <comment xml:lang="es">documento VRML</comment>
    <comment xml:lang="eu">VRML dokumentua</comment>
    <comment xml:lang="fi">VRML-asiakirja</comment>
    <comment xml:lang="fr">document VRML</comment>
    <comment xml:lang="hu">VRML-dokumentum</comment>
    <comment xml:lang="it">Documento VRML</comment>
    <comment xml:lang="ja">VRML ドキュメント</comment>
    <comment xml:lang="ko">VRML 문서</comment>
    <comment xml:lang="lt">VRML dokumentas</comment>
    <comment xml:lang="ms">Dokumen VRML</comment>
    <comment xml:lang="nb">VRML-dokument</comment>
    <comment xml:lang="nl">VRML-document</comment>
    <comment xml:lang="nn">VRML-dokument</comment>
    <comment xml:lang="pl">Dokument VRML</comment>
    <comment xml:lang="pt">documento VRML</comment>
    <comment xml:lang="pt_BR">Documento VRML</comment>
    <comment xml:lang="ru">документ VRML</comment>
    <comment xml:lang="sq">Dokument VRML</comment>
    <comment xml:lang="sr">VRML документ</comment>
    <comment xml:lang="sv">VRML-dokument</comment>
    <comment xml:lang="uk">Документ VRML</comment>
    <comment xml:lang="vi">Tài liệu VRML</comment>
    <comment xml:lang="zh_CN">VRML 文档</comment>
    <comment xml:lang="zh_TW">VRML 文件</comment>
    <glob pattern="*.wrl"/>

    <!-- Gzip compressed -->
    <glob pattern="*.wrz"/>
    <glob pattern="*.wrl.gz"/>
  </mime-type>

  <mime-type type="image/x-3ds">
    <comment>3DS model</comment>
    <comment xml:lang="az">3D Studio rəsmi</comment>
    <comment xml:lang="bg">Изображение — 3D Studio</comment>
    <comment xml:lang="ca">imatge de 3D Studio</comment>
    <comment xml:lang="cs">Obrázek 3D Studio</comment>
    <comment xml:lang="cy">Delwedd "3D Studio"</comment>
    <comment xml:lang="da">3D Studio-billede</comment>
    <comment xml:lang="de">3D Studio-Bild</comment>
    <comment xml:lang="el">εικόνα 3D Studio</comment>
    <comment xml:lang="en_GB">3D Studio image</comment>
    <comment xml:lang="eo">bildo de 3D Studio</comment>
    <comment xml:lang="es">imagen de 3D Studio</comment>
    <comment xml:lang="eu">3D Studio-ko irudia</comment>
    <comment xml:lang="fi">3D Studio -kuva</comment>
    <comment xml:lang="fr">image 3D Studio</comment>
    <comment xml:lang="hu">3D Studio-kép</comment>
    <comment xml:lang="it">Immagine 3D Studio</comment>
    <comment xml:lang="ja">3D Studio 画像</comment>
    <comment xml:lang="ko">3D Studio 그림</comment>
    <comment xml:lang="lt">3D Studio paveikslėlis</comment>
    <comment xml:lang="ms">Imej 3D Studio</comment>
    <comment xml:lang="nb">3D Studio-bilde</comment>
    <comment xml:lang="nl">3D-Studio-afbeelding</comment>
    <comment xml:lang="nn">3D Studio-bilete</comment>
    <comment xml:lang="pl">Obraz 3D Studio</comment>
    <comment xml:lang="pt">imagem 3D Studio</comment>
    <comment xml:lang="pt_BR">Imagem do 3D Studio</comment>
    <comment xml:lang="ru">изображение 3D Studio</comment>
    <comment xml:lang="sq">Figurë 3D Studio</comment>
    <comment xml:lang="sr">3D Studio слика</comment>
    <comment xml:lang="sv">3D Studio-bild</comment>
    <comment xml:lang="uk">Зображення 3D Studio</comment>
    <comment xml:lang="vi">Ảnh xuởng vẽ 3D</comment>
    <comment xml:lang="zh_CN">3D Studio 图像</comment>
    <comment xml:lang="zh_TW">3D Studio 圖片</comment>
    <glob pattern="*.3ds"/>
  </mime-type>

  <!-- X3D mimes just written based on X3D encodings specs.

       Note that I add Gzip compressed extensions here.
       All X3D encoding specifications say that
       Gzip compressed files indeed have the same MIME type as
       the normal uncompressed files.

       The one problem with this is that this makes "sub-class-of"
       somewhat invalid: Gzip compressed X3D is not really a subclass
       of text/xml, it's a binary data now (unless it's Ok?).
       But, since the spec says the mime type should be same, there's
       no correct solution here. Simply removing "sub-class-of" clauses
       would suck (and cause warnings from nautilus when double-clicking
       the file, since detected by "file" contents would conflict with
       mime specified, so the "file type" description would switch
       between "X3D model" and "XML document").
  -->

  <mime-type type="model/x3d+vrml">
    <sub-class-of type="text/plain"/>
    <comment>X3D model (classic VRML encoding)</comment>
    <glob pattern="*.x3dv"/>
    <!-- Gzip compressed -->
    <glob pattern="*.x3dv.gz"/>
    <glob pattern="*.x3dvz"/>
  </mime-type>

  <mime-type type="model/x3d+xml">
    <sub-class-of type="text/xml"/>
    <comment>X3D model (XML encoding)</comment>
    <glob pattern="*.x3d"/>
    <!-- Gzip compressed -->
    <glob pattern="*.x3d.gz"/>
    <glob pattern="*.x3dz"/>
  </mime-type>

  <mime-type type="model/x3d+binary">
    <comment>X3D model (binary compressed)</comment>
    <glob pattern="*.x3db"/>
    <!-- Gzip compressed -->
    <glob pattern="*.x3db.gz"/>
  </mime-type>

  <!-- MIME type from http://en.wikipedia.org/wiki/COLLADA -->
  <mime-type type="model/vnd.collada+xml">
    <sub-class-of type="text/xml"/>
    <comment>COLLADA model</comment>
    <glob pattern="*.dae"/>
  </mime-type>

  <mime-type type="application/x-inventor">
    <sub-class-of type="text/plain"/>
    <comment>Inventor model</comment>
    <glob pattern="*.iv"/>
  </mime-type>

  <!-- MIME type invented by Kambi.
       Neither http://en.wikipedia.org/wiki/MD3_%28file_format%29
       nor http://filext.com/file-extension/md3 specifies any Mime type.
       Please report if there is any standard about this. -->
  <mime-type type="application/x-md3">
    <comment>MD3 (Quake 3 engine) model</comment>
    <glob pattern="*.md3"/>
  </mime-type>

  <!-- MIME type invented by Kambi.
       http://en.wikipedia.org/wiki/Wavefront_.obj_file says to use
       text/plain, but that's useless for us, we need a unique MIME type
       to handle this.
       Please report if there is any standard about this. -->
  <mime-type type="application/x-wavefront-obj">
    <sub-class-of type="text/plain"/>
    <comment>Wavefront OBJ model</comment>
    <glob pattern="*.obj"/>
  </mime-type>

  <!-- MIME type invented by Kambi.
       Please report if there is any standard about this. -->
  <mime-type type="application/x-geo">
    <sub-class-of type="text/plain"/>
    <comment>Videoscape GEO model</comment>
    <glob pattern="*.geo"/>
  </mime-type>

  <!-- MIME type invented by Michalis Kamburelis.
       castle-anim-frames is my format,
       see [https://castle-engine.io/castle_animation_frames.php],
       so we can invent any mime type we want here. -->
  <mime-type type="application/x-castle-anim-frames">
    <sub-class-of type="text/xml"/>
    <comment>Castle Game Engine animation</comment>
    <glob pattern="*.kanim"/>
    <glob pattern="*.castle-anim-frames"/>
  </mime-type>

  <!-- view3dscene also handles Spine animations in *.json,
       although we cannot create MIME type that captures Spine animations
       and not other json content... -->
  <mime-type type="application/json">
    <sub-class-of type="text/plain"/>
    <comment>JSON document</comment>
    <glob pattern="*.json"/>
  </mime-type>

  <mime-type type="model/gltf+json">
    <comment>glTF</comment>
    <glob pattern="*.gltf"/>
  </mime-type>

  <mime-type type="model/gltf-binary">
    <comment>glTF Binary</comment>
    <glob pattern="*.glb"/>
  </mime-type>
</mime-info>
```

## update-mime-database代码

### 下载代码

找个目录，然后`apt source shared-mime-info`, 里面只有几个文件，`find . -name update-mime-database.c`得到结果`./shared-mime-info-2.2/src/update-mime-database.c
`。

打开它，`write_cache`函数是它写入cache文件的地方，

```
	{
		FILE *stream;
		char *path;
		
		path = g_strconcat(mime_dir, "/mime.cache.new", NULL);
		stream = fopen_gerror(path, error);
		if (!stream)
			goto out;
		write_cache(stream);
		if (!fclose_gerror(stream, error))
			goto out;
		if (!atomic_update(path, error))
			goto out;
		g_free(path);
	}

```

写的cache文件的名字为`mime.cache`,存放在:

```
~/.local/share/mime/
/usr/share/mime
```


它们是二进制可映射到内存为map的文件格式，所以查询速度非常的快。


## kbuildsycoca

kbuildsycoca是kservice的一部分，代码可以在[这里](https://github.com/KDE/kservice/tree/master/src/kbuildsycoca)看。


[这里](https://web.fe.up.pt/~jmcruz/etc/kde/kdeqt/kde3arch/ksycoca.html)有一个文档，介绍了sycoca的用法。
sycoca的意思是system configuration cache，三个单词各取了两个字母。
