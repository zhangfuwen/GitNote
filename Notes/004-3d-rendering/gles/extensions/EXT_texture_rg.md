# EXT_texture_rg

Name

    EXT_texture_rg

Name Strings
    
    GL_EXT_texture_rg

Contributors

    Contributors to ARB_texture_rg, on which this extension is based
    Kyle Haughey
    Richard Schreyer

Contact

    Benj Lipchak, Apple (lipchak 'at' apple.com)

Status 
    
    Complete

Version
    
    Date: July 22, 2011
    Revision: 3

Number

    OpenGL ES Extension #103

Dependencies

    Requires OpenGL ES 2.0.

    Written based on the wording of the OpenGL ES 2.0.25 Full Specification
    (November 2, 2010).

    OES_texture_float affects the definition of this extension.

    OES_texture_half_float affects the definition of this extension.

    APPLE_framebuffer_multisample affects the definition of this extension.

Overview

    Historically one- and two-component textures have been specified in OpenGL
    ES using the luminance or luminance-alpha (L/LA) formats. With the advent
    of programmable shaders and render-to-texture capabilities these legacy
    formats carry some historical artifacts which are no longer useful.

    For example, when sampling from such textures, the luminance values are
    replicated across the color components. This is no longer necessary with
    programmable shaders.
    
    It is also desirable to be able to render to one- and two-component format
    textures using capabilities such as framebuffer objects (FBO), but
    rendering to L/LA formats is under-specified (specifically how to map
    R/G/B/A values to L/A texture channels).

    This extension adds new base internal formats for one-component RED and
    two-component RG (red green) textures as well as sized RED and RG internal
    formats for renderbuffers. The RED and RG texture formats can be used for
    both texturing and rendering into with framebuffer objects.

New Procedures and Functions
    
    None

New Tokens

    Accepted by the <internalformat> parameter of TexImage2D and CopyTexImage2D,
    and the <format> parameter of TexImage2D, TexSubImage2D, and ReadPixels:

        RED_EXT                 0x1903
        RG_EXT                  0x8227

    Accepted by the <internalformat> parameter of RenderbufferStorage and
    RenderbufferStorageMultisampleAPPLE:

        R8_EXT                  0x8229
        RG8_EXT                 0x822B

Additions to Chapter 2 of the OpenGL ES 2.0 Specification (OpenGL ES Operation)

    None

Additions to Chapter 3 of the OpenGL ES 2.0 Specification (Rasterization)

    (Add the following to Table 3.3: "TexImage2D and ReadPixels formats")

    Format Name     Element Meaning and Order      Target Buffer
    -----------     -------------------------      -------------
    RED_EXT         R                              Color
    RG_EXT          R, G                           Color

    (Add the following to Table 3.4: "Valid pixel format and type combinations")
    (as modified by OES_texture_float and OES_texture_half_float)

    Format          Type                           Bytes per Pixel
    -----------     -------------------------      ---------------
    RED_EXT         FLOAT                          4
    RED_EXT         HALF_FLOAT_OES                 2
    RED_EXT         UNSIGNED_BYTE                  1
    RG_EXT          FLOAT                          8
    RG_EXT          HALF_FLOAT_OES                 4
    RG_EXT          UNSIGNED_BYTE                  2

    (Add the following to Table 3.8: "Conversion from RGBA and depth pixel
    components to internal texture")

    Base Internal Format     RGBA       Internal Components
    --------------------     ------     -------------------
    RED_EXT                  R          R
    RG_EXT                   R,G        R,G

    (Modify Table 3.9: "CopyTexImage internal format/color buffer combinations")

                            Texture Format
    Color Buffer      A  L  LA  R  RG  RGB  RGBA
    ------------      -  -  --  -  --  ---  ----
    A                 X
    R                    X      X
    RG                   X      X  X
    RGB                  X      X  X   X
    RGBA              X  X  X   X  X   X    X

    (Add the following to Table 3.12: "Correspondence of filtered texture
    components to texture source color components")

    Texture Base        Texture source color
    Internal Format     C_s             A_s
    ---------------     -------------   ------
    RED_EXT             (R_t, 0, 0)     1
    RG_EXT              (R_t, G_t, 0)   1

Additions to Chapter 4 of the OpenGL ES 2.0 Specification (Per-Fragment
Operations and the Framebuffer)

    In section 4.3.1 "Reading Pixels", subsection "Obtaining Pixels from the
    Framebuffer", modify the last sentence to read:

    "If the framebuffer does not support G, B, or A values then the G, B, and A
    values that are obtained are 0.0, 0.0, and 1.0 respectively."

    In section 4.4.5 "Framebuffer Completeness", modify the last sentence of
    the second paragraph to read:

    "Color-renderable formats contain red, and possibly green, blue, and alpha
    components; depth-renderable formats contain depth components; and
    stencil-renderable formats contain stencil components."

    (Add the following to Table 4.5: "Renderbuffer image formats, showing their
    renderable type (color-, depth-, or stencil-renderable) and the number of
    bits each format contains for color (R, G, B, A), depth (D), and stencil
    (S) components")

    Sized Internal  Renderable Type   R bits G bits B bits A bits D bits S bits
    Format
    --------------  ----------------  ------ ------ ------ ------ ------ ------
    R8_EXT        color-renderable  8
    RG8_EXT       color-renderable  8      8

Additions to Chapter 5 of the OpenGL ES 2.0 Specification (Special Functions)

    None

Additions to Chapter 6 of the OpenGL ES 2.0 Specification (State and State
Requests)

    None

Dependencies on OES_texture_float

    If OES_texture_float is not supported, then omit the rows of
    Table 3.4 that have Type FLOAT.

Dependencies on OES_texture_half_float

    If OES_texture_half_float is not supported, then omit the rows of 
    Table 3.4 that have Type HALF_FLOAT_OES.

Dependencies on APPLE_framebuffer_multisample

    If APPLE_framebuffer_multisample is not supported, then all references to
    RenderbufferStorageMultisampleAPPLE should be ignored.

Revision History
    
    #1 February 22, 2011, khaughey
        - initial version adapted from ARB_texture_rg.
    #2 June 16, 2011, benj
        - add interaction with APPLE_framebuffer_multisample
    #3 July 22, 2011, benj
        - rename from APPLE to EXT
