# 


Name
    
    EXT_texture_lod_bias

Name Strings

    GL_EXT_texture_lod_bias

Notice

    Copyright NVIDIA Corporation, 1999, 2000.

Contact

    Mark Kilgard, NVIDIA (mjk 'at' nvidia.com)

Status

    Shipping since late 1999.

    The texture LOD bias functionality in OpenGL 1.4 is based on this
    extension though the OpenGL 1.4 functionality added the ability to
    specify a second per-texture object bias term.  The OpenGL 1.4 enum
    values match the EXT enum values.

Version

    Last Modified Date: June 23, 2009

Number

    OpenGL Extension #186
    OpenGL ES Extension #60

Dependencies

    Written based on the wording of the OpenGL 1.2 specification.

    Affects ARB_multitexture.
    
    Interacts with OpenGL ES 1.1.

Overview

    OpenGL computes a texture level-of-detail parameter, called lambda
    in the GL specification, that determines which mipmap levels and
    their relative mipmap weights for use in mipmapped texture filtering.

    This extension provides a means to bias the lambda computation
    by a constant (signed) value.  This bias can provide a way to blur
    or pseudo-sharpen OpenGL's standard texture filtering.

    This blurring or pseudo-sharpening may be useful for special effects
    (such as depth-of-field effects) or image processing techniques
    (where the mipmap levels act as pre-downsampled image versions).
    On some implementations, increasing the texture lod bias may improve
    texture filtering performance (at the cost of texture bluriness).

    The extension mimics functionality found in Direct3D.

Issues

    Should the texture LOD bias be settable per-texture object or
    per-texture stage?

      RESOLUTION:  Per-texture stage.  This matches the Direct3D
      semantics for texture lod bias.  Note that this differs from
      the semantics of SGI's SGIX_texture_lod_bias extension that
      has the biases per-texture object.

      This also allows the same texture object to be used by two different
      texture units for different blurring.  This is useful for
      extrapolating detail between various levels of detail in a
      mipmapped texture.

      For example, you can extrapolate texture detail with
      ARB_multitexture and EXT_texture_env_combine by computing

        (B0 - B2) * 2 + B2

      where B0 is a non-biased texture (normal sharpness) and B2 is
      the same texture but bias by 2 levels-of-detail (fairly blurry).
      This has the effect of increasing the high-frequency information
      in the texture.  There are immediate Earth Sciences and medical
      imaging applications for this technique.

      Per-texture stage control of the LOD bias is also useful for
      allowing an application to control overall texture bluriness.
      This can be used in games to simulate disorientation (note that
      only textures will blur, not edges).  It can also be used to
      globally control texturing performance.  An application may be
      able to sustain a constant frame rate by avoiding texture fetch
      stalls by using slightly blurrier textures.

    How does EXT_texture_lod_bias differ from SGIX_texture_lod bias?

      EXT_texture_lod_bias adds a bias to lambda.  The
      SGIX_texture_lod_bias extension changes the computation of rho (the
      log2 of which is lambda).  The SGIX extension provides separate
      biases in each texture dimension.  The EXT extension does not
      provide an "directionality" in the LOD control.

    Does the texture lod bias occur before or after the TEXTURE_MAX_LOD
    and TEXTURE_MIN_LOD clamping?

      RESOLUTION:  BEFORE.  This allows the texture lod bias to still
      be clamped within the max/min lod range.

    Does anything special have to be said to keep the biased lambda value
    from being less than zero or greater than the maximum number of
    mipmap levels?

      RESOLUTION:  NO.  The existing clamping in the specification
      handles these situations.

    The texture lod bias is specified to be a float.  In practice, what
    sort of range is assumed for the texture lod bias?

      RESOLUTION:  The MAX_TEXTURE_LOD_BIAS_EXT implementation constant
      advertises the maximum absolute value of the supported texture
      lod bias.  The value is recommended to be at least the maximum
      mipmap level supported by the implementation.

    The texture lod bias is specified to be a float.  In practice, what
    sort of precision is assumed for the texture lod bias?

      RESOLUTION;  This is implementation dependent.  Presumably,
      hardware would implement the texture lod bias as a fractional bias
      but the exact fractional precision supported is implementation
      dependent.  At least 4 fractional bits is recommended.

New Procedures and Functions

    None

New Tokens

    Accepted by the <target> parameters of GetTexEnvfv, GetTexEnviv,
    TexEnvi, TexEnvf, Texenviv, and TexEnvfv:

        TEXTURE_FILTER_CONTROL_EXT          0x8500

    When the <target> parameter of GetTexEnvfv, GetTexEnviv, TexEnvi,
    TexEnvf, TexEnviv, and TexEnvfv is TEXTURE_FILTER_CONTROL_EXT, then
    the value of <pname> may be:
        
        TEXTURE_LOD_BIAS_EXT                0x8501

    Accepted by the <pname> parameters of GetBooleanv, GetIntegerv,
    GetFloatv, and GetDoublev:

        MAX_TEXTURE_LOD_BIAS_EXT            0x84FD

Additions to Chapter 2 of the 1.2 Specification (OpenGL Operation)

     None

Additions to Chapter 3 of the 1.2 Specification (Rasterization)

 --  Section 3.8.5 "Texture Minification"
 
     Change the first formula under "Scale Factor and Level of Detail" to read:

     "The choice is governed by a scale factor p(x,y), the level of detail
     parameter lambda(x,y), defined as

                 lambda'(x,y) = log2[p(x,y)] + lodBias

     where lodBias is the texture unit's (signed) texture lod bias parameter
     (as described in Section 3.8.9) clamped between the positive and negative
     values of the implementation defined constant MAX_TEXTURE_LOD_BIAS_EXT."

 --  Section 3.8.9 "Texture Environments and Texture Functions"

     Change the first paragraph to read:

     "The command

        void TexEnv{if}(enum target, enum pname, T param);
        void TexEnv{if}v(enum target, enum pname, T params);

     sets parameters of the texture environment that specifies how texture
     values are interepreted when texturing a fragment or sets per-texture
     unit texture filtering parameters.  The possible target parameters
     are TEXTURE_ENV or TEXTURE_FILTER_CONTROL_EXT.  ...  When target is
     TEXTURE_ENV, the possible environment parameters are TEXTURE_ENV_MODE
     and TEXTURE_ENV_COLOR. ... When target is TEXTURE_FILTER_CONTROL_EXT,
     the only possible texture filter parameter is TEXTURE_LOD_BIAS_EXT.
     TEXTURE_LOD_BIAS_EXT is set to a signed floating point value that
     is used to bias the level of detail parameter, lambda, as described
     in Section 3.8.5."

     Add a final paragraph at the end of the section:

     "The state required for the per-texture unit filtering parameters
     consists of one floating-point value."

Additions to Chapter 4 of the 1.2 Specification (Per-Fragment Operations
and the Frame Buffer)

     None

Additions to Chapter 5 of the 1.2 Specification (Special Functions)

     None

Additions to Chapter 6 of the 1.2 Specification (State and State Requests)

 --  Section 6.1.3 "Texture Environments and Texture Functions"

     Change the third sentence of the third paragraph to read:

     "The env argument to GetTexEnv must be either TEXTURE_ENV or
     TEXTURE_FILTER_CONTROL_EXT."

Additions to the GLX Specification

     None

Dependencies on OpenGL ES 1.1

     If the GL is OpenGL ES 1.1, omit reference to GetDoublev.

Errors

     INVALID_ENUM is generated when TexEnv is called with a <pname> of
     TEXTURE_FILTER_CONTROL_EXT and the value of <param> or what is pointed to
     by <params> is not TEXTURE_LOD_BIAS_EXT.

New State

(table 6.14, p204) add the entry:

Get Value                 Type   Get Command  Initial Value     Description      Sec     Attribute
-----------------------   ----   -----------  --------------    ---------------  -----   ---------
TEXTURE_LOD_BIAS_EXT      R      GetTexEnvfv  0.0               Biases texture   3.8.9    texture
                                                                level of detail

(When ARB_multitexture is supported, the TEXTURE_LOD_BIAS_EXT state is per-texture unit.)

New Implementation State

(table 6.24, p214) add the following entries:

Get Value                    Type    Get Command   Minimum Value   Description         Sec     Attribute
--------------------------   ----    -----------   -------------   -----------------   ------  --------------
MAX_TEXTURE_LOD_BIAS_EXT     R+      GetFloatv     4.0             Maximum             3.8.9   -
                                                                   absolute texture
                                                                   lod bias

Revision History

    6/23/09 (Jon Leech) - assign OpenGL ES extension number.

    4/29/09 (Benj Lipchak) - add interaction with OpenGL ES 1.1.

    8/27/03 - updated status to mention OpenGL 1.4 functionality.

    8/26/03 - fixed incorrect enum name (TEXTURE_FILTER_CONTROL_EXT is
    correct) in the Errors section.

    6/2/00 - add spec language to allow GetTexEnv to accept
    TEXTURE_FILTER_CONTROL_EXT.

