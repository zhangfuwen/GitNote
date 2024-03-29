# ANGLE_d3d_share_handle_client_buffer

Name

    ANGLE_d3d_share_handle_client_buffer

Name Strings

    EGL_ANGLE_d3d_share_handle_client_buffer

Contributors

    John Bauman
    Alastair Patrick
    Daniel Koch

Contacts

    John Bauman, Google Inc. (jbauman 'at' chromium.org)

Status

    Complete
    Implemented (ANGLE r650)

Version

    Version 3, May 12, 2011

Number

    EGL Extension #38

Dependencies

    Requires the EGL_ANGLE_surface_d3d_texture_2d_share_handle extension.

    This extension is written against the wording of the EGL 1.4
    Specification.

Overview

    This extension allows creating EGL surfaces from handles to textures
    shared from the Direct3D API or from
    EGL_ANGLE_surface_texture_2d_share_handle.

New Types

    None

New Procedures and Functions

    None

New Tokens

    Accepted in the <buftype> parameter of eglCreatePbufferFromClientBuffer:

        EGL_D3D_TEXTURE_2D_SHARE_HANDLE_ANGLE            0x3200

Additions to Chapter 3 of the EGL 1.4 Specification (EGL Functions and Errors)

    Replace the last sentence of paragraph 1 of Section 3.5.3 with the
    following text.
    "Currently, the only client API resources which may be bound in this
    fashion are OpenVG VGImage objects and Direct3D share handles."

    Replace the last sentence of paragraph 2 ("To bind a client API...") of
    Section 3.5.3 with the following text.
    "When <buftype> is EGL_OPENVG_IMAGE, the width and height of the pbuffer
    are determined by the width and height of <buffer>. When <buftype> is
    EGL_D3D_TEXTURE_2D_SHARE_HANDLE_ANGLE, the width and height are specified
    using EGL_WIDTH and EGL_HEIGHT, or else they default to zero. The width
    and height must match the dimensions of the texture which the share handle 
    was created from or else an EGL_BAD_ALLOC error is generated."

    Replace the third paragraph of Section 3.5.3 with the following text.
    "<buftype> specifies the type of buffer to be bound. The only allowed values
    of <buftype> are EGL_OPENVG_IMAGE and
    EGL_D3D_TEXTURE_2D_SHARE_HANDLE_ANGLE".

    Append the following text to the fourth paragraph of Section 3.5.3.
    "When <buftype> is EGL_D3D_TEXTURE_2D_SHARE_HANDLE_ANGLE, <buffer> must be
    a valid D3D share handle, cast into the type EGLClientBuffer. The handle
    may be obtained from the Direct3D9Ex CreateTexture function, from DXGI's
    GetSharedHandle method on an ID3D10Texture2D, or from the
    EGL_ANGLE_surface_d3d_texture_2d_share_handle extension."

Issues

Revision History

    Version 3, 2011/05/12
      - publish

    Version 2, 2011/05/03
      - specify EGL_D3D_TEXTURE_2D_SHARE_HANDLE
      - specify error if dimensions don't match

    Version 1, 2011/04/12 - first draft.
