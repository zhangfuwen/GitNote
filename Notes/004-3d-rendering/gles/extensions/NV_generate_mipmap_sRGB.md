# NV_generate_mipmap_sRGB

Name

    NV_generate_mipmap_sRGB

Name Strings

    GL_NV_generate_mipmap_sRGB

Contributors

    Contributors to EXT_sRGB
    Contributors to EXT_texture_sRGB

Contact

    Mathias Heyer, NVIDIA (mheyer 'at' nvidia.com

Status

    Complete.

Version

    Date: Sept 14, 2012

Number

    OpenGL ES Extension #144

Dependencies

    This extension requires OpenGL ES 1.0 or greater.  It is written based on
    the wording of the OpenGL ES 2.0.25 (November 2nd 2010) specification.

    This extension requires EXT_sRGB.

Overview

    EXT_sRGB requires GenerateMipmap() to throw INVALID_OPERATION on textures
    with sRGB encoding. NV_generate_mipmap_sRGB lifts this restriction.

New Procedures and Functions

    None

New Tokens

    None

Additions to Chapter 3 of the OpenGL ES2.0 Specification (Rasterization)

Modify Section 3.7.11: Mipmap Generation

    Remove the following sentence from the end of this section:

    "If the format of a texture is sRGB, the error INVALID_OPERATION is
    generated."

Errors

    Relaxation of INVALID_OPERATION errors
    ---------------------------------

    GenerateMipmap does not throw an error if the format of the texture is sRGB


New Implementation Dependent State

    None


Issues

     1) How should mipmap generation work for sRGB textures?

        RESOLVED:  The best way to perform mipmap generation for sRGB
        textures is by downsampling the sRGB image in a linear color
        space.

        This involves converting the RGB components of sRGB texels
        in a given texture image level to linear RGB space, filtering
        appropriately in that linear RGB space, and then converting the
        linear RGB values to sRGB for storage in the downsampled texture
        level image.

        (Remember alpha, when present, is linear even in sRGB texture
        formats.)

        The OpenGL specification says "No particular filter algorithm
        is required, though a box filter is recommended as the default
        filter" meaning there is no requirement for how even non-sRGB
        mipmaps should be generated.  So while the resolution to this
        issue is technically a recommendation, it is however a strongly
        advised recommendation.

        The rationale for why sRGB textures should be converted to
        linear space prior to filtering and converted back to sRGB after
        filtering is clear.  If an implementation naively simply performed
        linear filtering on (non-linear) sRGB components as if they were
        in a linear space, the result tends to be a subtle darkening of
        the texture images as mipmap generation continues recursively.
        This darkening is an inappropriate basis that the resolved
        "best way" above would avoid.


Revision History
    #01    9/14/2012    Mathias Heyer     First draft.
