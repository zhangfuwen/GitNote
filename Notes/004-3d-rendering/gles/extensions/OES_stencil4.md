# OES_stencil4

Name

    OES_stencil4

Name Strings

    GL_OES_stencil4

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

    Last Modifed Date: July 18, 2005

Number

    OpenGL ES Extension #32    

Dependencies

    OpenGL ES 1.0 is required.

    OES_framebuffer_object is required

Overview

    This extension enables 4-bit stencil component as a valid
    render buffer storage format.

Issues

 
New Tokens

    Accepted by the <internalformat> parameter of RenderbufferStorageOES:

        STENCIL_INDEX4_OES                 0x8D47 

New Procedures and Functions

    None.

Errors

    None.

New State

    None.

Revision History

    7/18/2005    Aaftab Munshi    Created the extension
