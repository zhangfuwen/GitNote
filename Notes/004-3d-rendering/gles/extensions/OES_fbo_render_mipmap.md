# OES_fbo_render_mipmap

Name

    OES_fbo_render_mipmap

Name Strings

    GL_OES_fbo_render_mipmap

Contact

    Aaftab Munshi (amunshi@ati.com)

Notice

    Copyright (c) 2005-2013 The Khronos Group Inc. Copyright terms at
        http://www.khronos.org/registry/speccopyright.html

Specification Update Policy

    Khronos-approved extension specifications are updated in response to
    issues and bugs prioritized by the Khronos OpenGL ES Working Group. For
    extensions which have been promoted to a core Specification, fixes will
    first appear in the latest version of that core Specification, and will
    eventually be backported to the extension document. This policy is
    described in more detail at
        https://www.khronos.org/registry/OpenGL/docs/update_policy.php

IP Status

    None.

Status

    Ratified by the Khronos BOP, July 22, 2005.

Version

    Last Modified Date:  July 6, 2005

Number

    OpenGL ES Extension #27

Dependencies

    OpenGL ES 1.0 is required.

    OES_framebuffer_object is required.

Overview

    OES_framebuffer_object allows rendering to the base level of a 
    texture only.  This extension removes this limitation by 
    allowing implementations to support rendering to any mip-level
    of a texture(s) that is attached to a framebuffer object(s).

    If this extension is supported, FramebufferTexture2DOES, and
    FramebufferTexture3DOES can be used to render directly into 
    any mip level of a texture image

Issues

 
New Tokens

    None.

New Procedures and Functions

    None.

Errors

    None.

New State

    None.

Revision History

    7/6/2005    Aaftab Munshi    Created the extension
