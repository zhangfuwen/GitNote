# OES_texture_cube_map

Name

    OES_texture_cube_map

Name Strings

    GL_OES_texture_cube_map

Contact

    Benj Lipchak (benj.lipchak 'at' amd.com)

Notice

    Copyright (c) 2007-2013 The Khronos Group Inc. Copyright terms at
        http://www.khronos.org/registry/speccopyright.html

Specification Update Policy

    Khronos-approved extension specifications are updated in response to
    issues and bugs prioritized by the Khronos OpenGL ES Working Group. For
    extensions which have been promoted to a core Specification, fixes will
    first appear in the latest version of that core Specification, and will
    eventually be backported to the extension document. This policy is
    described in more detail at
        https://www.khronos.org/registry/OpenGL/docs/update_policy.php

Status

    Ratified by the Khronos BOP

Version

    Version 2, April 16, 2015

Number

    OpenGL ES Extension #20

Dependencies

    OpenGL ES 1.0 is required.

    This extension is based on the ARB_texture_cube_map extension specification.

Overview

    This extension provides a new texture generation scheme for cube
    map textures.  Instead of the current texture providing a 1D, 2D,
    or 3D lookup into a 1D, 2D, or 3D texture image, the texture is a
    set of six 2D images representing the faces of a cube.  The (s,t,r)
    texture coordinates are treated as a direction vector emanating from
    the center of a cube.  At texture generation time, the interpolated
    per-fragment (s,t,r) selects one cube face 2D image based on the
    largest magnitude coordinate (the major axis).  A new 2D (s,t) is
    calculated by dividing the two other coordinates (the minor axes
    values) by the major axis value.  Then the new (s,t) is used to
    lookup into the selected 2D texture image face of the cube map.

    Unlike a standard 1D, 2D, or 3D texture that have just one target,
    a cube map texture has six targets, one for each of its six 2D texture
    image cube faces.  All these targets must be consistent, complete,
    and have equal width and height (ie, square dimensions).

    This extension also provides two new texture coordinate generation modes
    for use in conjunction with cube map texturing.  The reflection map
    mode generates texture coordinates (s,t,r) matching the vertex's
    eye-space reflection vector.  The reflection map mode
    is useful for environment mapping without the singularity inherent
    in sphere mapping.  The normal map mode generates texture coordinates
    (s,t,r) matching the vertex's transformed eye-space
    normal.  The normal map mode is useful for sophisticated cube
    map texturing-based diffuse lighting models.

    The intent of the new texgen functionality is that an application using
    cube map texturing can use the new texgen modes to automatically
    generate the reflection or normal vectors used to look up into the
    cube map texture.

    An application note:  When using cube mapping with dynamic cube
    maps (meaning the cube map texture is re-rendered every frame),
    by keeping the cube map's orientation pointing at the eye position,
    the texgen-computed reflection or normal vector texture coordinates
    can be always properly oriented for the cube map.  However if the
    cube map is static (meaning that when view changes, the cube map
    texture is not updated), the texture matrix must be used to rotate
    the texgen-computed reflection or normal vector texture coordinates
    to match the orientation of the cube map.  The rotation can be
    computed based on two vectors: 1) the direction vector from the cube
    map center to the eye position (both in world coordinates), and 2)
    the cube map orientation in world coordinates.  The axis of rotation
    is the cross product of these two vectors; the angle of rotation is
    the arcsin of the dot product of these two vectors.

Issues

    Please refer to the ARB_texture_cube_map extension specification.

New Procedures and Functions

        void glTexGenfOES(GLenum coord, GLenum pname, GLfloat param);
        void glTexGenfvOES(GLenum coord, GLenum pname, const GLfloat *params);
        void glTexGeniOES(GLenum coord, GLenum pname, GLint param);
        void glTexGenivOES(GLenum coord, GLenum pname, const GLint *params);
        void glTexGenxOES(GLenum coord, GLenum pname, GLfixed param);
        void glTexGenxvOES(GLenum coord, GLenum pname, const GLfixed *params);

        void glGetTexGenfvOES(GLenum coord, GLenum pname, GLfloat *params);
        void glGetTexGenivOES(GLenum coord, GLenum pname, GLint *params);
        void glGetTexGenxvOES(GLenum coord, GLenum pname, GLfixed *params);

New Tokens

    Accepted by the <pname> parameter of TexGenfOES, TexGeniOES, TexGenxOES,
    TexGenfvOES, TexGenivOES, TexGenxvOES, GetTexGenfvOES, GetTexGenivOES, and
    GetTexGenxvOES:

        TEXTURE_GEN_MODE_OES                0x2500

    Accepted by the <params> parameter of TexGenfOES, TexGeniOES, TexGenxOES,
    TexGenfvOES, TexGenivOES, and TexGenxvOES when <pname> parameter is
    TEXTURE_GEN_MODE_OES:

        NORMAL_MAP_OES                      0x8511
        REFLECTION_MAP_OES                  0x8512

    Accepted by the <cap> parameter of Enable, Disable, IsEnabled, by the
    <pname> parameter of GetBooleanv, GetIntegerv, GetFloatv, and GetFixedv, and
    by the <target> parameter of BindTexture, GetTexParameterfv, 
    GetTexParameteriv, GetTexParameterxv, TexParameterf, TexParameteri,
    TexParameterx, TexParameterfv, TexParameteriv, and TexParameterxv:

        TEXTURE_CUBE_MAP_OES                0x8513

    Accepted by the <cap> parameter of Enable, Disable, IsEnabled, by the
    <pname> parameter of GetBooleanv, GetIntegerv, GetFloatv, and GetFixedv, and
    by the <coord> parameter of TexGenfOES, TexGeniOES, TexGenxOES, TexGenfvOES, 
    TexGenivOES, TexGenxvOES, GetTexGenfvOES, GetTexGenivOES, and
    GetTexGenxvOES:

        TEXTURE_GEN_STR_OES                 0x8D60

    Accepted by the <pname> parameter of GetBooleanv, GetIntegerv,
    GetFloatv, and GetDoublev:

        TEXTURE_BINDING_CUBE_MAP_OES        0x8514

    Accepted by the <target> parameter of TexImage2D, CopyTexImage2D, 
    TexSubImage2D, and CopySubTexImage2D:

        TEXTURE_CUBE_MAP_POSITIVE_X_OES     0x8515
        TEXTURE_CUBE_MAP_NEGATIVE_X_OES     0x8516
        TEXTURE_CUBE_MAP_POSITIVE_Y_OES     0x8517
        TEXTURE_CUBE_MAP_NEGATIVE_Y_OES     0x8518
        TEXTURE_CUBE_MAP_POSITIVE_Z_OES     0x8519
        TEXTURE_CUBE_MAP_NEGATIVE_Z_OES     0x851A

    Accepted by the <pname> parameter of GetBooleanv, GetDoublev,
    GetIntegerv, and GetFloatv:

        MAX_CUBE_MAP_TEXTURE_SIZE_OES       0x851C

Additions to Chapter 2 of the OpenGL 1.5 Specification (OpenGL Operation)

 --  Section 2.11.4 "Generating Texture Coordinates"

      Change the last sentence in the 1st paragraph (page 37) to:

      "If <pname> is TEXTURE_GEN_MODE_OES, then either <params> points to
      or <param> is an integer that is one of the symbolic constants
      REFLECTION_MAP_OES, or NORMAL_MAP_OES."  OBJECT_LINEAR, EYE_LINEAR, 
      and SPHERE_MAP texture coordinate generation modes are not supported.

      Add these paragraphs after the 4th paragraph (page 38):

      "If TEXTURE_GEN_MODE_OES indicates REFLECTION_MAP_OES, compute the
      reflection vector r as described for the SPHERE_MAP mode.  Then the
      value assigned to an s coordinate is s = rx; the value assigned to a t
      coordinate is t = ry; and the value assigned to a r coordinate is r = rz.

      If TEXTURE_GEN_MODE_OES indicates NORMAL_MAP_OES, compute the normal
      vector nf as described in section 2.10.3.  Then the value assigned
      to an s coordinate is s = nfx; the value assigned to a t coordinate is
      t = nfy; and the value assigned to a r coordinate is r = nfz.  (The values
      nfx, nfy, and nfz are the components of nf.)

      A texture coordinate generation function is enabled or disabled
      using Enable and Disable with an argument of TEXTURE_GEN_STR_OES.  
      TEXTURE_GEN_S, TEXTURE_GEN_T, TEXTURE_GEN_R and TEXTURE_GEN_Q
      argument values to Enable and Disable are not supported.

      The last paragraph's last sentence (page 38) should be changed to:

      "Initially all texture generation modes are set to REFLECTION_MAP_OES"

Additions to Chapter 3 of the 1.5 Specification (Rasterization)

 --  Section 3.8.1 "Texture Image Specification"

     Change the second and third to last sentences on page 116 to:

     "<target> must be one of TEXTURE_2D for a 2D texture, or one of
     TEXTURE_CUBE_MAP_POSITIVE_X_OES, TEXTURE_CUBE_MAP_NEGATIVE_X_OES,
     TEXTURE_CUBE_MAP_POSITIVE_Y_OES, TEXTURE_CUBE_MAP_NEGATIVE_Y_OES,
     TEXTURE_CUBE_MAP_POSITIVE_Z_OES, or TEXTURE_CUBE_MAP_NEGATIVE_Z_OES
     for a cube map texture."

     Add the following paragraphs after the first paragraph on page 117:

     "A 2D texture consists of a single 2D texture image.  A cube
     map texture is a set of six 2D texture images.  The six cube map
     texture targets form a single cube map texture though each target
     names a distinct face of the cube map.  The TEXTURE_CUBE_MAP_*_OES
     targets listed above update their appropriate cube map face 2D
     texture image.  Note that the six cube map 2D image tokens such as
     TEXTURE_CUBE_MAP_POSITIVE_X_OES are used when specifying, updating,
     or querying one of a cube map's six 2D image, but when enabling
     cube map texturing or binding to a cube map texture object (that is
     when the cube map is accessed as a whole as opposed to a particular
     2D image), the TEXTURE_CUBE_MAP_OES target is specified.

     When the target parameter to TexImage2D is one of the six cube map
     2D image targets, the error INVALID_VALUE is generated if the width
     and height parameters are not equal.

     If cube map texturing is enabled at the time a primitive is
     rasterized and if the set of six targets are not "cube complete",
     then it is as if texture mapping were disabled.  The targets of
     a cube map texture are "cube complete" if the array 0 of all six
     targets have identical, positive, and square dimensions, the array
     0 of all six targets were specified with the same internalformat,
     and the array 0 of all six targets have the same border width."

     After the 14th paragraph (page 116) add:

     "In a similiar fashion, the maximum allowable width and height
     (they must be the same) of a cube map texture must be at least
     2^(k-lod) for image arrays level 0 through k, where k is the
     log base 2 of MAX_CUBE_MAP_TEXTURE_SIZE_OES."

 --  Section 3.8.2 "Alternate Texture Image Specification Commands"

     Update the second paragraph (page 120) to say:

     ... "Currently, <target> must be
     TEXTURE_2D, TEXTURE_CUBE_MAP_POSITIVE_X_OES,
     TEXTURE_CUBE_MAP_NEGATIVE_X_OES, TEXTURE_CUBE_MAP_POSITIVE_Y_OES,
     TEXTURE_CUBE_MAP_NEGATIVE_Y_OES, TEXTURE_CUBE_MAP_POSITIVE_Z_OES,
     or TEXTURE_CUBE_MAP_NEGATIVE_Z_OES." ...

     Add after the second paragraph (page 120), the following:

     "When the target parameter to CopyTexImage2D is one of the six cube
     map 2D image targets, the error INVALID_VALUE is generated if the
     width and height parameters are not equal."

 --  Section 3.8.3 "Texture Parameters"

     Change paragraph one (page 124) to say:

     ... "<target> is the target, either TEXTURE_2D or TEXTURE_CUBE_MAP_OES."

     Add a final paragraph saying:

     "Texture parameters for a cube map texture apply to cube map
     as a whole; the six distinct 2D texture images use the
     texture parameters of the cube map itself.

 --  Section 3.8.5 "Texture Minification" under "Mipmapping"

     Change the first full paragraph on page 130 to:

     ... "If texturing is enabled for two-dimensional texturing but not cube map
     texturing (and TEXTURE_MIN_FILTER is one that requires a mipmap) at the
     time a primitive is rasterized and if the set of arrays
     TEXTURE_BASE_LEVEL through q = min{p,TEXTURE_MAX_LEVEL} is incomplete,
     based on the dimensions of array 0, then it is as if texture mapping were
     disabled."

     Follow the first full paragraph on page 130 with:

     "If cube map texturing is enabled and TEXTURE_MIN_FILTER is one that
     requires mipmap levels at the time a primitive is rasterized and
     if the set of six targets are not "mipmap cube complete", then it
     is as if texture mapping were disabled.  The targets of a cube map
     texture are "mipmap cube complete" if the six cube map targets are
     "cube complete" and the set of arrays TEXTURE_BASE_LEVEL through
     q are not incomplete (as described above)."

 --  Section 3.8.7 "Texture State and Proxy State"

     Change the first sentence of the first paragraph (page 131) to say:

     "The state necessary for texture can be divided into two categories.
     First, there are the nine sets of mipmap arrays (one each for the
     one-, two-, and three-dimensional texture targets and six for the
     cube map texture targets) and their number." ...

     Change the second paragraph (page 132) to say:

     "In addition to the one-, two-, three-dimensional, and the six cube
     map sets of image arrays, the partially instantiated one-, two-,
     and three-dimensional and one cube map sets of proxy image arrays
     are maintained." ...

 --  Section 3.8.8 "Texture Objects"

     Change the first sentence of the first paragraph (page 132) to say:

     "In addition to the default textures TEXTURE_2D and TEXTURE_CUBE_MAP_OES,
     named two-dimensional texture objects and cube map texture objects can be
     created and operated on." ...

     Change the second paragraph (page 132) to say:

     "A texture object is created by binding an unused name to
     TEXTURE_2D or TEXTURE_CUBE_MAP_OES." ...
     "If the new texture object is bound to TEXTURE_2D or TEXTURE_CUBE_MAP_OES,
     it remains a two-dimensional or cube map texture until it is deleted."

     Change the third paragraph (page 133) to say:

     "BindTexture may also be used to bind an existing texture object to
     either TEXTURE_2D or TEXTURE_CUBE_MAP_OES."

     Change paragraph five (page 133) to say:

     "In the initial state, TEXTURE_2D and TEXTURE_CUBE_MAP_OES have two-
     dimensional and cube map state vectors associated with them respectively."
     ...  "The initial two-dimensional and cube map texture is therefore
     operated upon, queried, and applied as TEXTUER_2D and TEXTURE_CUBE_MAP_OES
     respectively while 0 is bound to the corresponding targets."

     Change paragraph six (page 133) to say:

     ... "If a texture that is currently bound to one of the targets TEXTURE_2D
     or TEXTURE_CUBE_MAP_OES is deleted, it is as though BindTexture has been
     executed with the same <target> and <texture> zero." ...

 --  Section 3.8.10 "Texture Application"

     Replace the beginning sentences of the first paragraph (page 138)
     with:

     "Texturing is enabled or disabled using the generic Enable
     and Disable commands, respectively, with the symbolic constants
     TEXTURE_2D or TEXTURE_CUBE_MAP_OES to enable the two-dimensional or cube
     map texturing respectively.  If the cube map texture and the two-
     dimensional texture are enabled, then cube map texturing is used.  If
     texturing is disabled, a rasterized fragment is passed on unaltered to the
     next stage of the GL (although its texture coordinates may be discarded).
     Otherwise, a texture value is found according to the parameter values of
     the currently bound texture image of the appropriate dimensionality.

     However, when cube map texturing is enabled, the rules are
     more complicated.  For cube map texturing, the (s,t,r) texture
     coordinates are treated as a direction vector (rx,ry,rz) emanating
     from the center of a cube.  (The q coordinate can be ignored since
     it merely scales the vector without affecting the direction.) At
     texture application time, the interpolated per-fragment (s,t,r)
     selects one of the cube map face's 2D image based on the largest
     magnitude coordinate direction (the major axis direction).  If two
     or more coordinates have the identical magnitude, the implementation
     may define the rule to disambiguate this situation.  The rule must
     be deterministic and depend only on (rx,ry,rz).  The target column
     in the table below explains how the major axis direction maps to
     the 2D image of a particular cube map target.

      major axis
      direction     target                             sc     tc    ma
      ----------    -------------------------------    ---    ---   ---
       +rx          TEXTURE_CUBE_MAP_POSITIVE_X_OES    -rz    -ry   rx
       -rx          TEXTURE_CUBE_MAP_NEGATIVE_X_OES    +rz    -ry   rx
       +ry          TEXTURE_CUBE_MAP_POSITIVE_Y_OES    +rx    +rz   ry
       -ry          TEXTURE_CUBE_MAP_NEGATIVE_Y_OES    +rx    -rz   ry
       +rz          TEXTURE_CUBE_MAP_POSITIVE_Z_OES    +rx    -ry   rz
       -rz          TEXTURE_CUBE_MAP_NEGATIVE_Z_OES    -rx    -ry   rz

     Using the sc, tc, and ma determined by the major axis direction as
     specified in the table above, an updated (s,t) is calculated as
     follows

        s   =   ( sc/|ma| + 1 ) / 2
        t   =   ( tc/|ma| + 1 ) / 2

     This new (s,t) is used to find a texture value in the determined
     face's 2D texture image using the rules given in sections 3.8.5
     and 3.8.6." ...

Additions to Chapter 4 of the 1.2 Specification (Per-Fragment Operations
and the Frame Buffer)

     None

Additions to Chapter 5 of the 1.2 Specification (Special Functions)

Additions to Chapter 6 of the 1.2 Specification (State and State Requests)

 --  Section 6.1.3 "Enumerated Queries"

     Change the fourth paragraph (page 183) to say:

     "The GetTexParameter parameter <target> may be one of TEXTURE_2D,
     or TEXTURE_CUBE_MAP_OES, indicating the currently bound two-dimensional
     or cube map texture object."

Additions to the GLX Specification

     None

Errors

     INVALID_VALUE is generated when the target parameter to TexImage2D
     or CopyTexImage2D is one of the six cube map 2D image targets and
     the width and height parameters are not equal.

New State

(table 6.12, p202) add the following entries:

Get Value                        Type    Get Command   Initial Value   Description           Sec    Attribute
---------                        ----    -----------   -------------   -----------           ------ --------------
TEXTURE_CUBE_MAP_OES             B       IsEnabled     False           True if cube map      3.8.10 texture/enable
                                                                       texturing is enabled
TEXTURE_BINDING_CUBE_MAP_OES     Z+      GetIntegerv   0               Texture object        3.8.8  texture
                                                                       for TEXTURE_CUBE_MAP
TEXTURE_CUBE_MAP_POSITIVE_X_OES  nxI     N/A           see 3.8         positive x face       3.8    -
                                                                       cube map texture
                                                                       image at lod i
TEXTURE_CUBE_MAP_NEGATIVE_X_OES  nxI     N/A           see 3.8         negative x face       3.8    -
                                                                       cube map texture
                                                                       image at lod i
TEXTURE_CUBE_MAP_POSITIVE_Y_OES  nxI     N/A           see 3.8         positive y face       3.8    -
                                                                       cube map texture
                                                                       image at lod i
TEXTURE_CUBE_MAP_NEGATIVE_Y_OES  nxI     N/A           see 3.8         negative y face       3.8    -
                                                                       cube map texture
                                                                       image at lod i
TEXTURE_CUBE_MAP_POSITIVE_Z_OES  nxI     N/A           see 3.8         positive z face       3.8    -
                                                                       cube map texture
                                                                       image at lod i
TEXTURE_CUBE_MAP_NEGATIVE_Z_OES  nxI     N/A           see 3.8         negative z face       3.8    -
                                                                       cube map texture
                                                                       image at lod i

(table 6.14, p204) add the following entries:

Get Value            Type Get Command    Initial Value      Description Sec     Attribute
---------            ---- -----------    -------------      ----------- ------  ---------
TEXTURE_GEN_MODE_OES Z2   GetTexGenivOES REFLECTION_MAP_OES Function used for   2.10.4 texture
                                                            texgen (for s,t,r)
TEXTURE_GEN_STR_OES  B    IsEnabled      FALSE              True if texgen is   2.10.4 texture
                                                            enabled (for s,t,r)

New Implementation Dependent State

(table 6.24, p214) add the following entry:

Get Value                       Type    Get Command   Minimum Value   Description           Sec    Attribute
---------                       ----    -----------   -------------   -----------           ------ --------------
MAX_CUBE_MAP_TEXTURE_SIZE_OES   Z+      GetIntegerv   16              Maximum cube map      3.8.1  -
                                                                      texture image
                                                                      dimension

Revision History

Version 1, November 8, 2007 (Benj Lipchak) - First version cleaned up for ES
Version 2, April 16, 2015 (Jon Leech) - Remove border width term bt (nonexistent in ES)
