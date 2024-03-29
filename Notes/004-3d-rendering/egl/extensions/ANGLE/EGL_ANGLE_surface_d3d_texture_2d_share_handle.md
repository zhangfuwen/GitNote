# ANGLE_surface_d3d_texture_2d_share_handle

Name

    ANGLE_surface_d3d_texture_2d_share_handle

Name Strings

    EGL_ANGLE_surface_d3d_texture_2d_share_handle

Contributors

    Vladimir Vukicevic
    Daniel Koch

Contacts

    Vladimir Vukicevic (vladimir 'at' pobox.com)

Status

    Complete
    Implemented (ANGLE r558)

Version

    Version 2, December 21, 2010

Number

    EGL Extension #29

Dependencies

    Requires the EGL_ANGLE_query_surface_pointer extension.

    This extension is written against the wording of the EGL 1.4
    Specification.

Overview

    Some EGL implementations generate EGLSurface handles that are
    backed by Direct3D 2D textures.  For such surfaces, a D3D share
    handle can be generated, allowing access to the same surface
    from the Direct3D API.

New Types

    None

New Procedures and Functions

    None

New Tokens

    Accepted in the <attribute> parameter of eglQuerySurfacePointerANGLE:

        EGL_D3D_TEXTURE_2D_SHARE_HANDLE_ANGLE            0x3200

Additions to Chapter 3 of the EGL 1.4 Specification (EGL Functions and Errors)

    Add to table 3.5, "Queryable surface attributes and types":

        Attribute                              Type      Description
        ---------                              ----      -----------
        EGL_D3D_TEXTURE_2D_SHARE_HANDLE_ANGLE  pointer   Direct3D share handle

    Add before the last paragraph in section 3.5, "Surface attributes":

   "Querying EGL_D3D_TEXTURE_2D_SHARE_HANDLE_ANGLE returns a Direct3D
    share handle, or NULL if a share handle for the surface is not
    available.  The share handle must be queried using
    eglQuerySurfaceAttribPointerANGLE.  Before using a Direct3D surface
    created with this share handle, ensure that all rendering
    to the EGLSurface with EGL client APIs has completed.

    The Direct3D share handle may be passed as the pSharedHandle
    parameter of the Direct3D9Ex CreateTexture function, or via the
    Direct3D10 OpenSharedResource function.  If used with Direct3D 9,
    the level argument to CreateTexture must be 1, and the dimensions
    must match the dimensions of the EGL surface.  If used with
    Direct3D 10, OpenSharedResource should be called with the
    ID3D10Texture2D uuid to obtain an ID3D10Texture2D object.

Issues

Revision History

    Version 3, 2011/02/11 - publish

    Version 2, 2010/12/21
      - renamed token to EGL_D3D_TEXTURE_2D_SHARE_HANDLE_ANGLE (adding "2D")
      - renamed extension to ANGLE_surface_d3d_texture_2d_share_handle
      - added language about supported usage of the shared handle from D3D

    Version 1, 2010/12/07 - first draft.
