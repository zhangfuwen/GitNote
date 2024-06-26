# EXT_external_objects_fd

## 简述

从fd导入texture特别有用。 fd导入后就不能再操作了，这个函数有ownership转移意味。

顺序是：fd -> memory object -> texture

方法：

```cpp

void importFdAsTexture(int fd, int size, GLuint tex, int w, int h) {
    GLuint tex;
    glGenTextures(1, &tex);
    // create memObj and import fd as memObj
    GLuint memObj = 0;
    glCreateMemoryObjectsEXT(1, &memObj);
    GLint param = GL_TRUE;
    glMemoryObjectParameterivEXT(memObj, GL_DEDICATED_MEMORY_OBJECT_EXT, &param);
    glImportMemoryFdEXT(memObj, size, GL_HANDLE_TYPE_OPAQUE_FD_EXT, fd);

    // make memOjb texture's storage
    glTextureStorageMem2DEXT(tex, 1, (GLuint)GL_RGBA8, w, h, memObj, 0);

    return tex;
}

```

上述代码中，fd和size是要导入的fd和它内容的大小，w, h是要导入为的texture的name和宽高。

## 用到的几个函数的介绍

### glCreateMemoryObjectsEXT

所属扩展：[EXT_external_objects](http://www.xjbcode.fun/Notes/004-3d-rendering/gles/extensions/EXT_external_objects.html)

原型：

```cpp
void glCreateMemoryObjectsEXT(sizei n,
                                uint *memoryObjects);
```     

语法跟glGenTextures类似.

### glMemoryObjectParameterivEXT

所属扩展：[EXT_external_objects](http://www.xjbcode.fun/Notes/004-3d-rendering/gles/extensions/EXT_external_objects.html)

原型：

```cpp
   void glMemoryObjectParameterivEXT(uint memoryObject,
                                    enum pname,
                                    const int *params);

```                                 

语法跟glTexParameteri类似。

pname可取的值为：

`DEDICATED_MEMORY_OBJECT_EXT                0x9581`

[[ The following are available when GL_EXT_protected_textures is
       available ]]

`PROTECTED_MEMORY_OBJECT_EXT                0x959B`

可取的值：

    | Name                        | Legal Values |
    +-----------------------------+--------------+
    | DEDICATED_MEMORY_OBJECT_EXT | FALSE, TRUE  |
    | PROTECTED_MEMORY_OBJECT_EXT | FALSE, TRUE  |
    +-----------------------------+--------------+


### glImportMemoryFdEXT

所属扩展：[EXT_external_objects_fd](http://www.xjbcode.fun/Notes/004-3d-rendering/gles/extensions/EXT_external_objects_fd.html)

原型：

```cpp
void ImportMemoryFdEXT(uint memory,
                       uint64 size,
                       enum handleType,
                       int fd);

```

其为memory为memObj的name，即`glCreateMemoryObjectsEXT`的出参。

`handleType`目前可取值只有`HANDLE_TYPE_OPAQUE_FD_EXT`。

`size`的单位是字节,byte。

### glTextureStorageMem2DEXT

所属扩展:[EXT_external_objects](http://www.xjbcode.fun/Notes/004-3d-rendering/gles/extensions/EXT_external_objects.html)

原型：

```cpp
  void TextureStorageMem2DEXT(uint texture,
                                sizei levels,
                                enum internalFormat,
                                sizei width,
                                sizei height,
                                uint memory,
                                uint64 offset);

```

texture和memory分别为texture和memobj的name。
levels是说导入的纹理有几个mipmap level。
offset是说在memobj中的偏移，正常为0。
internalFormat为纹理的内部格式。

# 正文

Name

    EXT_external_objects_fd

Name Strings

    GL_EXT_memory_object_fd
    GL_EXT_semaphore_fd

Contributors

    Carsten Rohde, NVIDIA
    James Jones, NVIDIA
    Jan-Harald Fredriksen, ARM
    Jeff Juliano, NVIDIA

Contact

    James Jones, NVIDIA (jajones 'at' nvidia.com)

Status

    Complete

Version

    Last Modified Date: June 2, 2017
    Revision: 7

Number

    504
    OpenGL ES Extension #281

Dependencies

    Written against the OpenGL 4.5 and OpenGL ES 3.2 specifications

    GL_EXT_memory_object_fd requires GL_EXT_memory_object

    GL_EXT_semaphore_fd requires GL_EXT_semaphore

    Requires ARB_texture_storage or a version of OpenGL or OpenGL ES that
    incorporates it.

Overview

    Building upon the OpenGL memory object and semaphore framework
    defined in EXT_external_objects, **this extension enables an OpenGL
    application to import a memory object or semaphore from POSIX file
    descriptor external handles**.

New Procedures and Functions

    If the GL_EXT_memory_object_fd string is reported, the following
    commands are added:

    void ImportMemoryFdEXT(uint memory,
                           uint64 size,
                           enum handleType,
                           int fd);

    If the GL_EXT_semaphore_fd string is reported, the following commands
    are added:

    void ImportSemaphoreFdEXT(uint semaphore,
                              enum handleType,
                              int fd);


New Tokens

    The following tokens are added if either of the GL_EXT_memory_object_fd
    or GL_EXT_semaphore_fd strings are reported:

    Accepted by the <handleType> parameter of ImportMemoryFdEXT() or
    ImportSemaphoreFdEXT().

        HANDLE_TYPE_OPAQUE_FD_EXT                  0x9586

Additions to Chapter 4 of the OpenGL 4.5 Specification (Event Model)

    Add the following entry to table 4.2 "Commands for importing
    external semaphore handles."

        | Handle Type               | Import command       |
        +---------------------------+----------------------+
        | HANDLE_TYPE_OPAQUE_FD_EXT | ImportSemaphoreFdEXT |
        +---------------------------+----------------------+

    Replace the paragraph in section 4.2.1 beginning "External handles
    are often defined..." with the following

        The command

            ImportSemaphoreFdEXT(uint semaphore,
                                 enum handleType,
                                 int fd);

        imports a semaphore from the file descriptor <fd>.  What type of
        object <fd> refers to is determined by <handleType>.  A successful
        import operation transfers ownership of <fd> to the GL
        implementation, and performing any operation on <fd> in the
        application after an import results in undefined behavior.

Additions to Chapter 6 of the OpenGL 4.5 Specification (Memory Objects)

    Add the following entry to table 6.2 "Commands for importing
    external memory handles."

        | Handle Type               | Import command    |
        +---------------------------+-------------------+
        | HANDLE_TYPE_OPAQUE_FD_EXT | ImportMemoryFdEXT |
        +---------------------------+-------------------+

    Replace the paragraph in section 6.1 beginning "External handles are
    often defined..." with the following

        The command

            void ImportMemoryFdEXT(uint memory,
                                   uint64 size,
                                   enum handleType,
                                   int fd);

        imports a memory object of length <size> from the file descriptor
        <fd>.  What type of object <fd> refers to is determined by
        <handleType>.  A successful import operation transfers ownership
        of <fd> to the GL implementation, and performing any operation on
        <fd> in the application after an import results in undefined
        behavior.

Issues

    1)  Does this extension need to support importing Android/Linux
        sync FD handles?

        RESOLVED: No.  These are already usable in GL via extensions to the
        EGLSync mechanism.  Adding them here in order to support them in GLX
        contexts is not compelling enough to justify the additional effort.

Revision History

    Revision 7, 2017-06-02 (James Jones)
        - Added extension numbers.
        - Clarified which extensions each command and token belongs to.
        - Marked complete.

    Revision 6, 2017-05-24 (James Jones)
        - Filled in real token values

    Revision 5, 2017-04-04 (James Jones)
        - Clarified the effects of import operations on file descriptors.

    Revision 4, 2017-03-17 (James Jones)
        - Renamed from KHR to EXT.

    Revision 3, 2016-09-28 (James Jones)
        - Merged GL_EXT_memory_object_fd and GL_EXT_semaphore_fd.
        - Added spec body describing the new commands and tokens.
        - Added issue 1.

    Revision 2, 2016-08-15 (Jeff Juliano)
        - Clarified overview text.

    Revision 1, 2016-08-05 (James Jones)
        - Initial draft.
