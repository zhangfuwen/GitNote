# EXT_texture_sRGB_decode

Name

    EXT_texture_sRGB_decode

Name Strings

    GL_EXT_texture_sRGB_decode

Contributors

    Matt Collins, Apple Inc
    Alex Eddy, Apple Inc
    Mark Kilgard, NVIDIA
    Chris Niederauer, Apple Inc
    Richard Schreyer, Apple Inc
    Henri Verbeet, CodeWeavers
    Brent Wilson, NVIDIA
    Jeff Bolz, NVIDIA
    Dan Omachi, Apple Inc
    Jason Green, TransGaming
    Daniel Koch, NVIDIA
    Mathias Heyer, NVIDIA

Contact

    Matt Collins, Apple Inc (matthew.collins 'at' apple.com)

Status

    Shipping on OS X 10.7

Version

    Date: November 8, 2017
    Revision: 0.91

Number

    OpenGL Extension #402
    OpenGL ES Extension #152

Dependencies

    OpenGL 2.1 or EXT_texture_sRGB requried for OpenGL

    OpenGL ES 3.0 or EXT_sRGB are required for OpenGL ES

    OpenGL 3.0 or later interacts with this extension.

    OpenGL ES 2.0 interacts with this extension.

    OpenGL ES 3.0 interacts with this extension.

    ARB_bindless_texture interacts with this extension.

    ARB_sampler_objects interacts with this extension.

    ARB_framebuffer_object interacts with this extension.

    EXT_direct_state_access interacts with this extension.

    EXT_texture_compression_s3tc interacts with this extension.

    EXT_texture_integer interacts with this extension.

    EXT_sRGB interacts with this extension.

    NV_sRGB_formats interacts with this extension.

    NV_generate_mipmap_sRGB interacts with this extension.

    KHR_texture_compression_astc_ldr interacts with this extension.

    ETC2 texure compression formats interact with this extension.

    This extension is written against the OpenGL 2.1 (December 1, 2006)
    specification.

Overview

    The EXT_texture_sRGB extension (promoted to core in OpenGL 2.1)
    provides a texture format stored in the sRGB color space. Sampling one
    of these textures will always return the color value decoded into a
    linear color space. However, an application may wish to sample and
    retrieve the undecoded sRGB data from the texture and manipulate
    that directly.

    This extension adds a Texture Parameter and Sampler Object parameter to
    allow sRGB textures to be read directly, without decoding.

    The new parameter, TEXTURE_SRGB_DECODE_EXT controls whether the
    decoding happens at sample time. It only applies to textures with an
    internal format that is sRGB and is ignored for all other textures.
    This value defaults to DECODE_EXT, which indicates the texture
    should be decoded to linear color space.

New Procedures and Functions

    None

New Tokens

    Accepted by the <pname> parameter of TexParameterf, TexParameteri,
    TexParameterfv, TexParameteriv, TexParameterIiv, TexParameterIuiv,
    TexParameterIivEXT, TexParameterIuivEXT, TextureParameterfEXT,
    TextureParameterfvEXT, TextureParameteriEXT, TextureParameterivEXT,
    TextureParameterIivEXT, TextureParameterIuivEXT,
    MultiTexParameterfEXT, MultiTexParameterfvEXT, MultiTexParameteriEXT,
    MultiTexParameterivEXT, MultiTexParameterIivEXT,
    MultiTexParameterIuivEXT, GetTexParameterfv, GetTexParameteriv,
    GetTexParameterIiv, GetTexParameterIuiv, GetTexParameterIivEXT,
    GetTexParameterIuivEXT, GetTextureParameterfEXT,
    GetTextureParameterfvEXT, GetTextureParameteriEXT,
    GetTextureParameterivEXT, GetTextureParameterIivEXT,
    GetTextureParameterIuivEXT, GetMultiTexParameterfEXT,
    GetMultiTexParameterfvEXT, GetMultiTexParameteriEXT,
    GetMultiTexParameterivEXT, GetMultiTexParameterIivEXT,
    GetMultiTexParameterIuivEXT, SamplerParameteri, SamplerParameterf,
    SamplerParameteriv, SamplerParameterfv, SamplerParameterIiv,
    SamplerParameterIuiv, GetSamplerParameteriv, GetSamplerParameterfv,
    GetSamplerParameterIiv, and GetSamplerParameterIuiv:

        TEXTURE_SRGB_DECODE_EXT        0x8A48

    Accepted by the <param> parameter of TexParameterf, TexParameteri,
    TexParameterfv, TexParameteriv, TexParameterIiv, TexParameterIuiv,
    TexParameterIivEXT, TexParameterIuivEXT, TextureParameterfEXT,
    TextureParameterfvEXT, TextureParameteriEXT, TextureParameterivEXT,
    TextureParameterIivEXT, TextureParameterIuivEXT,
    MultiTexParameterfEXT, MultiTexParameterfvEXT, MultiTexParameteriEXT,
    MultiTexParameterivEXT, MultiTexParameterIivEXT,
    MultiTexParameterIuivEXT, SamplerParameteri, SamplerParameterf,
    SamplerParameteriv, SamplerParameterfv, SamplerParameterIiv, and
    SamplerParameterIuiv:

        DECODE_EXT             0x8A49
        SKIP_DECODE_EXT        0x8A4A

Additions to Chapter 3 of the 2.1 Specification (Rasterization)

    Add 1 new row to Table 3.18 (page 169).

    Name                       Type       Initial value     Legal values
    ----                       ----       -------------     ------------
    TEXTURE_SRGB_DECODE_EXT    enum        DECODE_EXT       DECODE_EXT, SKIP_DECODE_EXT

-- OpenGL: Section 3.8.8, Texture Minification

    Add to the end of the "Automatic Mipmap Generation" subsection:

    If the automatic mipmap generation is applied to a texture
    whose internal format is one of SRGB_EXT, SRGB8_EXT,
    SRGB_ALPHA_EXT, SRGB8_ALPHA8_EXT, SLUMINANCE_ALPHA_EXT,
    SLUMINANCE8_ALPHA8_EXT, SLUMINANCE_EXT, SLUMINANCE8_EXT,
    COMPRESSED_SRGB_EXT, COMPRESSED_SRGB_ALPHA_EXT,
    COMPRESSED_SLUMINANCE_EXT, COMPRESSED_SLUMINANCE_ALPHA_EXT,
    COMPRESSED_SRGB_S3TC_DXT1_EXT, COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT,
    COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT, or
    COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT, and the TEXTURE_SRGB_DECODE_EXT
    parameter for the current texture unit is DECODE_EXT, the RGB
    texel components are decoded to a linear components as described
    in section 3.8.15 prior to downsampling; then after downsampling,
    the linear components are re-encoded as sRGB in the following manner:

    If cl is the linear color component, then the corresponding sRGB
    encoded component is encoded as follows

             {  cl * 12.92,                  cl < 0.0031308
        cs = {
             {  1.055 * cl^0.41666 - 0.055,  cl >= 0.0031308

    If the automatic mipmap generation is applied to a texture whose
    internal format is one of the sRGB formats listed previously and
    the TEXTURE_SRGB_DECODE_EXT parameter for the texture object is
    SKIP_DECODE_EXT, the sRGB decode and encode steps are skipped during
    mipmap generation.

-- OpenGL:  Section 3.8.15, sRGB Color Decoding

    (section was previously titled sRGB Color Conversion)

    Replace current text with the following:

    If the currently bound texture's internal format is one
    of SRGB_EXT, SRGB8_EXT, SRGB_ALPHA_EXT, SRGB8_ALPHA8_EXT,
    SLUMINANCE_ALPHA_EXT, SLUMINANCE8_ALPHA8_EXT, SLUMINANCE_EXT,
    SLUMINANCE8_EXT, COMPRESSED_SRGB_EXT, COMPRESSED_SRGB_ALPHA_EXT,
    COMPRESSED_SLUMINANCE_EXT, COMPRESSED_SLUMINANCE_ALPHA_EXT,
    COMPRESSED_SRGB_S3TC_DXT1_EXT, COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT,
    COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT, or
    COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT, and the TEXTURE_SRGB_DECODE_EXT
    parameter for the current texture unit is DECODE_EXT, the red, green, and blue
    components are decoded from an sRGB color space to a linear color
    space as part of filtering described in sections 3.8.8 and 3.8.9.
    Any alpha component is left unchanged. Ideally, implementations
    should perform this color decoding on each sample prior to filtering
    but implementations are allowed to perform this decoding after
    filtering (though this post-filtering approach is inferior to
    decoding from sRGB prior to filtering).

    The decoding from an sRGB encoded component, cs, to a linear
    component, cl, is as follows

            {  0,                          cs <= 0
            {
            {  cs / 12.92,                 0 < cs <= 0.04045
       cl = {
            {  ((cs + 0.055)/1.055)^2.4,   0.04045 < cs < 1
            {
            {  1,                          cs >= 1

    Assume cs is the sRGB component in the range [0,1].

    If the TEXTURE_SRGB_DECODE_EXT parameter is SKIP_DECODE_EXT, the value
    is returned without decoding. The TEXTURE_SRGB_DECODE_EXT
    parameter state is ignored for any texture with an internal format
    not explicitly listed above, as no decoding needs to be done.

--- OpenGL ES 3.2: Section 8.21, sRGB Texture Color Conversion
--- OpenGL ES 3.0: Section 3.8.16, sRGB Texture Color Conversion
--- OpenGL ES 2.0: Section 3.7.14, sRGB Texture Color Conversion

    Add after the first paragraph of the section:

   "The conversion of sRGB color space components to linear color space is
    always applied if the TEXTURE_SRGB_DECODE_EXT parameter is DECODE_EXT.
    Table X.1 describes whether the conversion is skipped if the
    TEXTURE_SRGB_DECODE_EXT parameter is SKIP_DECODE_EXT, depending on
    the function used for the access, whether the access occurs through a
    bindless sampler, and whether the texture is statically accessed
    elsewhere with a texelFetch function."

    Add a new table X.1, Whether the conversion of sRGB color space
    components to linear color space is skipped when the
    TEXTURE_SRGB_DECODE_EXT parameter is SKIP_DECODE_EXT.

                                       texelFetch       other builtin
      --------------------------------------------------------------------
      non-bindless sampler,            n/a              yes
      no accesses with
      texelFetch

      non-bindless sampler,            no               undefined
      statically accessed with
      texelFetch

      bindless sampler                 undefined        yes

Dependencies on ARB_bindless_texture

    If ARB_bindless_texture is NOT supported, delete all references to
    bindless samplers.

Dependencies on ARB_sampler_objects or OpenGL 3.3 or later

    If ARB_sampler_objects or OpenGL 3.3 or later is NOT supported,
    delete all references to SamplerParameter* and GetSamplerParameter*.

Dependencies on ARB_framebuffer_object or OpenGL 3.0 or later

    If ARB_framebuffer_object or OpenGL 3.0 or later is supported, the
    explanation in the "Automatic Mipmap Generation" section applies to
    the GenerateMipmap command as well.

Dependencies on EXT_texture_compression_s3tc

    If EXT_texture_compression_s3tc is NOT supported, delete
    COMPRESSED_SRGB_S3TC_DXT1_EXT, COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT,
    COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT, and
    COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT from Section 3.8.15.

Dependencies on EXT_texture_integer

    If EXT_texture_integer is NOT supported, delete references to
    TexParameterIivEXT, TexParameterIuivEXT, GetTexParameterIivEXT,
    and GetTexParameterIuivEXT.

Dependencies on EXT_direct_state_access

    If EXT_direct_state_access is NOT supported, delete
    references to TextureParameterfEXT, TextureParameterfvEXT,
    TextureParameteriEXT, TextureParameterivEXT, TextureParameterIivEXT,
    TextureParameterIuivEXT, MultiTexParameterfEXT,
    MultiTexParameterfvEXT, MultiTexParameteriEXT, MultiTexParameterivEXT,
    MultiTexParameterIivEXT, MultiTexParameterIuivEXT,
    GetTextureParameterfEXT, GetTextureParameterfvEXT,
    GetTextureParameteriEXT, GetTextureParameterivEXT,
    GetTextureParameterIivEXT, GetTextureParameterIuivEXT,
    GetMultiTexParameterfEXT, GetMultiTexParameterfvEXT,
    GetMultiTexParameteriEXT, GetMultiTexParameterivEXT,
    GetMultiTexParameterIivEXT, and GetMultiTexParameterIuivEXT.

Dependencies on OpenGL 3.0

    If OpenGL 3.0 or later is NOT supported, delete references
    to TexParameterIiv, TexParameterIuiv, GetTexParameterIiv,
    and GetTexParameterIuiv.

Interactions with OpenGL ES

    If OpenGL ES 3.0 is NOT supported, delete references
    to TexParameterIiv, TexParameterIuiv, GetTexParameterIiv,
    and GetTexParameterIuiv, GetTexParameterfv, GetTexParameteriv.

    If OpenGL ES 3.0 or NV_generate_mipmap_sRGB is supported,
    TEXTURE_SRGB_DECODE_EXT will control the linearization of sRGB
    texture levels while generating the mipmap levels. The section
    "Automatic Mipmap Generation" applies to glGenerateMipmap instead.

    If neither OpenGL ES 3.0 nor NV_sampler_objects is supported,
    delete all references to SamplerParameter* and GetSamplerParameter*.

    If NV_sampler_objects is supported, substitue the ARB_sampler_objects
    references with corresponding commands of NV_sampler_objects.

Interactions with KHR_texture_compression_astc_ldr

    If KHR_texture_compression_astc_ldr is supported, the
    TEXTURE_SRGB_DECODE_EXT texture and/or sampler parameter affects the
    COMPRESSED_SRGB8_ALPHA8_ASTC_*_KHR formats as described in the Section
    3.8.16 edits.

Interactions with ETC2 compressed texture formats

    If the ETC2 texture compression formats (part of OpenGL ES 3.0 and OpenGL
    4.3) are supported, the TEXTURE_SRGB_DECODE_EXT texture and/or sampler
    parameter affects the COMPRESSED_SRGB8_*ETC2* formats as described in the
    Section 3.8.16 edits.

Interactions with NV_sRGB_formats

    If NV_sRGB_formats is supported, the TEXTURE_SRGB_DECODE_EXT texture and/or
    sampler parameter affects the new SRGB and SLUMINANCE formats as described
    in the Section 3.7.14 edits.

Errors

    INVALID_ENUM is generated if the <pname> parameter of
    TexParameter[i,f,Ii,Iui][v][EXT], MultiTexParameter[i,f,Ii,Iui][v]EXT,
    TextureParameter[i,f,Ii,Iui][v]EXT, SamplerParameter[i,f,Ii,Iui][v]
    is TEXTURE_SRGB_DECODE_EXT when the <param> parameter is not one of
    DECODE_EXT or SKIP_DECODE_EXT.

New State

    In table 6.20, Texture Objects, p. 384, add the following:

    Get Value                     Type  Get Command           Initial Value  Description       Sec.   Attribute
    ----------------------------  ----  --------------------  -------------  ----------------  -----  ---------
    TEXTURE_SRGB_DECODE_EXT       Z_2   GetTexParameter[if]v  DECODE_EXT     indicates when    3.8.15 texture
                                                                             sRGB textures
                                                                             are decoded from
                                                                             sRGB or the
                                                                             decoding step is
                                                                             skipped

    Add to Table 6.23 of ARB_sampler_objects, "Textures (state per sampler object)":

    Get Value                     Type  Get Command               Initial Value  Description       Sec.   Attribute
    ----------------------------  ----  ------------------------  -------------  ----------------  -----  ---------
    TEXTURE_SRGB_DECODE_EXT       Z_2   GetSamplerParameter[if]v  DECODE_EXT     indicates when    3.8.15 texture
                                                                                 sRGB textures
                                                                                 are decoded from
                                                                                 sRGB or the
                                                                                 decoding step is
                                                                                 skipped

Issues

    1) What should this extension be called?

        RESOLVED: EXT_texture_sRGB_decode

        The purpose of this extension is to allow developers to skip
        the sRGB-to-linear decoding detailed in Section 3.8.15.
        Since this is a decoding of the sRGB value into linear space, we
        use that word to describe the pname. The enum indicating this
        decoding is to happen is DECODE, as that is what the GL will do.
        The enum that indicates this decoding is to be skipped is then
        appropriately, SKIP_DECODE.

    2) Should this allow for filters other than NEAREST on undecoded
       sRGB values?

        RESOLVED: YES

        Hardware supports this, and it is left up to the programmer.

    3) Do we generate an error if TEXTURE_SRGB_DECODE_EXT is changed for normal
       textures?

        RESOLVED: NO

        This is similar to the ARB_shadow and ARB_framebuffer_sRGB extensions - the flag
        is ignored for non-sRGB texture internal formats.

    4) Should we add forward-looking support for ARB_sampler_objects?

        RESOLVED: YES

        If ARB_sampler_objects exists in the implementation, the sampler
        objects should also include this parameter per sampler.

    5) What is the expense of changing the sRGB-ness of a texture without
       this extension?

        RESOLVED:  If an application wants to use a texture with sRGB
        texels and then switch to using it with linear filtering (or vice
        versa), OpenGL without this extension requires the application
        to read back all the texels in all the mipmap levels of all the
        images, and respecify a different texture object with a different
        texture format.  This is very expensive.

        With this extension, the driver can simply change the underlying
        hardware texture format associated with the texture to perform
        sRGB conversion on filtering or not.  This is very inexpensive.

        However, note that the functionality of this extension can also
        be obtained using the more modern approach provided by
        ARB_texture_view (added to OpenGL 4.3) and OES_texture_view.

    6) Do any major games or game engines depend on the ability to
       change the sRGB-ness of textures?

        RESOLVED:  Yes, Valve's Source engine used by Half-Life 2,
        Counter-Strike: Source, and Left 4 Dead; and Unreal Engine 3
        games including Unreal Tournament 3 and BioShock.

        These games and their underlying engines repeatedly render linear
        color values into textures and then texture from the rendered
        textures with sRGB texture filtering.

    7) Why not simply allow changing whether a standard GL_RGBA8
       can perform an sRGB color space conversion for filtering?

        RESOLVED:  Allowing this would create a consistency problem.
        Why would the GL_TEXTURE_SRGB_DECODE_EXT parameter not
        apply to GL_RGB4 or GL_RGB12 textures too?  In practice,
        sRGB color conversion for texture filtering is only typically
        supported in hardware for a small subset of texture formats
        (corresponding to the sized internal formats introduced by the
        EXT_texture_sRGB specification).  It's essentially only 8-bit
        fixed-point unsigned textures where sRGB color conversion makes
        sense.  And the initial value of the parameter (GL_DECODE_EXT) would be
        appropriate for sRGB texture formats but not conventional linear
        texture formats (as no decoding needs to be done). Having the

        texture parameter apply just to sRGB texture eliminates the ambiguity
        of which conventional texture formats can and cannot have sRGB decoding
        applied to them. This also eliminates the burden of having every future

        texture format extension specify whether or not the sRGB decoding parameter
        applies to them.

        Direct3D 9 handles this situation by advertising for each surface
        format (which double as texture formats) a D3DUSAGE_QUERY_SRGBREAD
        parameter.  In practice, Direct3D 9 implementation only advertise
        the D3DUSAGE_QUERY_SRGBREAD parameter for 8-bit fixed-point
        unsigned RGB or luminance formats, corresponding to the formats
        available from EXT_texture_sRGB.

    8) Does there need to be a control for whether to update (and
       possibly blend) framebuffer pixels in sRGB or linear color space?

        RESOLVED:  The EXT_framebuffer_sRGB extension (made core in OpenGL
        3.0) already has this capability with the GL_FRAMEBUFFER_SRGB_EXT
        enable.

        The GL_FRAMEBUFFER_ATTACHMENT_COLOR_ENCODING query parameter
        roughly corresponds to Direct3D 9's D3DUSAGE_QUERY_SRGBWRITE
        parameter.

    9) How is the border color handled when toggling sRGB color
       conversion for blending of sRGB textures?

        RESOLVED:  Consistent with the EXT_texture_sRGB specification, the
        border color is always specified as a linear value (never sRGB).
        So changing the TEXTURE_SRGB_DECODE_EXT parameter will
        not affect the resulting sampled border color.

        If an implementation were to store the texture border color in a
        format consistent with the texel format (including the sRGB color
        space), this would require such implementations to convert the
        (linear) texture border RGB components to sRGB space.
        In this case, this would mean an implementation to re-specify
        the texture border color state in the hardware when the
        TEXTURE_SRGB_DECODE_EXT parameter for an sRGB texture
        changed.

        Alternatively, if the implementation stored the texture
        border color in texture formant-independent format (say 4
        floating-point values) and always treated this as a linear RGB
        color for purposes of texture filtering, no sRGB conversion
        of the texture border color would ever occur.  In this case,
        this would mean an implementation would NOT need to re-specify
        the texture border color state in the hardware when the
        TEXTURE_SRGB_DECODE_EXT parameter for an sRGB texture
        changed.

   10) How is mipmap generation of sRGB textures affected by the
       TEXTURE_SRGB_DECODE_EXT parameter?

        RESOLVED:  When the TEXTURE_SRGB_DECODE parameter is DECODE_EXT
        for an sRGB texture, mipmap generation should decode sRGB texels
        to a linear RGB color space, perform downsampling, then encode
        back to an sRGB color space.  (Issue 24 in the EXT_texture_sRGB
        specification provides a rationale for why.)  When the parameter
        is SKIP_DECODE_EXT instead, mipmap generation skips the encode
        and decode steps during mipmap generation.  By skipping the
        encode and decode steps, sRGB mipmap generation should match
        the mipmap generation for a non-sRGB texture.

        The TEXTURE_SRGB_DECODE_EXT texture parameter has no effect on
        mipmap generation of non-sRGB textures.

        Direct3D 10 and Direct3D 11 expect mipmap generation for sRGB
        textures to be "correctly done" (meaning sRGB decode samples,
        perform weighted average in linear space, then sRGB encode
        the result).

        Direct3D 9 expects to NOT perform sRGB-correct mipmap generation.
        Hence the ability to generate mipmaps from an sRGB texture
        where you skip the decode (and encode) during mipmap generation
        is important.

   11) Does automatic mipmap generation change the smaller mipmap levels
       when the TEXTURE_SRGB_DECODE texture parameter changes?

        RESOLVED:  No, automatic mipmap generation only happens when the
        base level is changed.

        This means if the TEXTURE_SRGB_DECODE parameter is changed from
        DECODE_EXT to SKIP_DECODE_EXT (or vice versa), the texels in the
        smaller mipmap levels are not modified.

        Use the glGenerateMipmap command to regenerate mipmaps explicitly
        to reflect a change in the TEXTURE_SRGB_DECODE parameter.

   12) How is this extension expected to be used for Direct3D 9 emulation?

        RESOLVED: Direct3D texture resources that are created with a
        format supporting either the SRGBREAD or SRGBWRITE capabilities
        should be created as an OpenGL texture object with an sRGB
        internal format.

        This means that normal "linear" RGBA8 textures for Direct3D 9
        should be created as GL_SRGB8_ALPHA8 textures so they can be used
        with samplers where the GL_TEXTURE_SRGB_DECODE_EXT parameter of
        the sampler (assuming ARB_sampler_objects) determines whether
        they operate as linear textures (the GL_SKIP_DECODE_EXT) state
        or sRGB textures (the GL_DECODE_EXT state).

        Example for a Direct3D9 CreateTexture with a D3DFMT_A8R8G8B8 format:

          glTexImage2D(GL_TEXTURE_2D, 0, GL_SRGB8_ALPHA8,
            width, height, border, GL_UNSIGNED_BYTE, GL_RGBA, texels);
          glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SRGB_DECODE_EXT, GL_SKIP_DECODE_EXT);

        Notice the texture is created with a GL_SRGB_ALPHA8 format but
        immediately switched to GL_SKIP_DECODE_EXT.  This is because:

        1a) The format needs to be GL_SRGBA_ALPHA so that when used with
            a sampler configured with the (initial) value of zero (false)
            for D3DSAMP_SRGBTEXTURE, the texture will be filtered without
            sRGB decode.

        1b) Likewise, when D3DSAMP_SRGBTEXTURE is true for a sampler, the
            texture needs to be decoded to sRGB for filtering.  In this case,
            the OpenGL translation would use:

              glSamplerParameteri(sampler, GL_TEXTURE_SRGB_DECODE_EXT, GL_DECODE_EXT);

        2)  Direct3D9's D3DUSAGE_AUTOGENMIPMAP does not generate mipmaps
            in sRGB space (but rather in linear space).

        When rendering into a surface in Direct3D 9 with the
        D3DRS_SRGBWRITEENABLE render state (set by SetRenderState) set to false,
        the pixel updates (including blending) need to operate with
        GL_FRAMEBUFFER_SRGB disabled.  So:

          glDisable(GL_FRAMEBUFFER_SRGB);

        Likewise when the D3DRS_SRGBWRITEENABLE render state is true,
        OpenGL should operate

          glEnable(GL_FRAMEBUFFER_SRGB);

        Any texture with an sRGB internal format (for example,
        GL_SRGB8_ALPHA8 for the internal format) will perform sRGB decode
        before blending and encode after blending.  This matches the Direct3D9
        semantics when D3DUSAGE_QUERY_SRGBWRITE is true of the resource format.

   13) How is this extension expected to be used for Direct3D 10 and
       11 emulation?

        RESOLVED:  Direct3D 10 and 11 support non-mutable formats for sRGB
        textures (matching the original behavior of EXT_texture_sRGB,
        unextended by this extension).  So the DXGI_FORMAT_*_SRGB
        formats are always decoded from sRGB to linear (and vice versa)
        as necessary.  Formats not suffixed with _SRGB are never decoded
        or encoded to sRGB.

        Direct3D 10 and 11 support "typeless" resources with resource views
        that can have different formats.  So you can create a texture
        with a format of DXGI_FORMAT_R8G8B8A8_TYPELESS and then create
        shader resource views with the DXGI_FORMAT_R8G8B8A8_UNORM and
        DXGI_FORMAT_R8G8B8A8_UNORM_SRGB formats on that. This is a much
        more generic approach to decoupling storage and interpretation in
        Direct3D 10 and 11.  However support for "typeless" resources
        and resource views is beyond the scope of this extension.

        These two questions from Microsoft's Direct3D 10 Frequently Asked
        Questions list provide helpful context:

            "Q:  Where did the D3DSAMP_SRGBTEXTURE state go?

            A: SRGB was removed as part of the sampler state and now
            is tied to the texture format. Binding an SRGB texture will
            result in the same sampling you would get if you specified
            D3DSAMP_SRGBTEXTURE in Direct3D 9.

            Q:  What are these new SRGB formats?

            A:  SRGB was removed as part of the sampler state and is
            now tied to the texture format. Binding an SRGB texture will
            result in the same sampling you would get if you specified
            D3DSAMP_SRGBTEXTURE in Direct3D 9."

        This means that normal "linear" textures, such as
        DXGI_FORMAT_R8G8B8A8_UNORM, would be created as GL_RGBA8 textures
        (not sRGB), so that they will always behave as linear textures
        (never sRGB-decoded).

        On the other hand, DXGI_FORMAT_R8G8B8A8_UNORM_SRGB would be
        created as GL_SRGB8_ALPHA8.  Its texture sRGB decode parameter
        would be left with the initial value of GL_DECODE_EXT.  Mipmap
        generation for a texture using DXGI_FORMAT_R8G8B8A8_UNORM_SRGB
        (GL_RGB8_ALPHA8 in OpenGL) would perform the proper sRGB decode
        and encode needed for automatic mipmap generation.

        In Direct3D 10 and 11 emulation, the GL_TEXTURE_SRGB_DECODE_EXT
        parameter of sampler objects would simply be left at its initial
        value of GL_DECODE_EXT.  Unlike Direct3D 9 where the sampler
        controlled sRGB decode (via the D3DSAMP_SRGBTEXTURE), that parameter
        is not present in Direct3D 10 and 11 (see the FAQ questions above).

        The conclusion of this issue's discussion is that Direct3D
        10 and 11 emulation software should simply ignore the
        EXT_texture_sRGB_decode extension.  This is to be expected
        because the EXT_texture_sRGB_decode extension is meant to match
        the legacy functionality of Direct3D 9.

   14) Why does Table X.1 show "Undefined", and why does it appear in
       different columns depending on whether bindless samplers are used
       or not?

        RESOLVED: Conceptually, TEXTURE_SRGB_DECODE_EXT is part of the
        sampler state and should therefore not apply to texelFetch. However,
        not all hardware has the required bit in the sampler state.

        With bindless samplers, texture handles are *always* statically
        accessed by texelFetch (because an application could choose to do so
        at any time), so applying the same rules as for non-bindless
        samplers would make the functionality provided in this extension
        useless.

Revision History

        Rev.    Date    Author    Changes
        ----  --------  --------  -------------------------------------
        0.91  11/08/17  nhaehnle  Add interaction with bindless textures
                                  (API issue #51)
        0.90  04/27/16  Jon Leech Add interaction with texelFetch builtins
                                  (Bug 14934)
        0.89  08/14/13  dkoch     Add interactions with ASTC/ETC2/NV_sRGB_formats
        0.88  07/24/13  mheyer    Add OpenGL ES interactions
        0.87  08/22/11  mjk       correction to issue #8 from Jason
        0.86  08/22/11  mjk       corrections from Daniel + more interactions
        0.85  08/13/11  mjk       corrections to issues from Jason and Henri
        0.84  08/05/11  mjk       New issues to explain Direct3D interactions;
                                  fix table Get Command.
        0.83  07/15/11  mjk       "current texture unit" -> "texture
                                  object" for mipmap generation
        0.82  04/12/11  mjk       Mipmap generation interactions.
        0.81  11/18/10  mattc     Fixed language in error section.
                                  Cleaned up which functions take which tokens.
        0.8   11/18/10  mattc     Added issues from EXT_texture_sRGB_decode

                                  for background info. Cleaned up layout.
        0.71  11/18/10  mattc     Adapted apple_texture_linearize_srgb into
                                  this specification.
