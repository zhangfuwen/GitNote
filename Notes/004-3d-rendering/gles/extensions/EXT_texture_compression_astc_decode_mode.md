# EXT_texture_compression_astc_decode_mode

Name

    EXT_texture_compression_astc_decode_mode

Name Strings

    GL_EXT_texture_compression_astc_decode_mode
    GL_EXT_texture_compression_astc_decode_mode_rgb9e5

Contact

    Jan-Harald Fredriksen (jan-harald.fredriksen 'at' arm.com)

Contributors

    Edvard Fielding
    Jan-Harald Fredriksen
    Jakob Fries
    Tom Olson
    Jorn Nystad

IP Status

    No known issues.

Status

    DRAFT

Version

    Version 4, January 23, 2017

Number

    OpenGL ES Extension #276

Dependencies

    OpenGL ES 3.0 is required.

    GL_KHR_texture_compression_astc_hdr, GL_KHR_texture_compression_astc_ldr,
    or GL_OES_texture_compression_astc is required.

    This extension is written based on the wording of the OpenGL ES 3.2
    specification and the GL_KHR_texture_compression_astc_hdr extension.

    This extension interacts with GL_KHR_texture_compression_astc_hdr.

Overview

    Adaptive Scalable Texture Compression (ASTC) is a texture compression
    technology that is exposed by existing extensions and specifications.

    The existing specifications require that low dynamic range (LDR)
    textures are decompressed to FP16 values per component. In many cases,
    decompressing LDR textures to a lower precision intermediate result gives
    acceptable image quality. Source material for LDR textures is typically
    authored as 8-bit UNORM values, so decoding to FP16 values adds little
    value. On the other hand, reducing precision of the decoded result
    reduces the size of the decompressed data, potentially improving texture
    cache performance and saving power.

    The goal of this extension is to enable this efficiency gain on existing
    ASTC texture data. This is achieved by giving the application the ability
    to select the decoding precision.

    Two decoding options are provided by
    GL_EXT_texture_compression_astc_decode_mode
     - Decode to FP16: This is the default, and matches the required behavior
       in existing APIs.
     - Decode to UNORM8: This is provided as an option in LDR mode.

    If GL_EXT_texture_compression_astc_decode_mode_rgb9e5 is supported, then
    a third decoding option is provided:
     - Decode to RGB9_E5: This is provided as an option in both LDR and HDR
       mode. In this mode, negative values cannot be represented and are
       clamped to zero. The alpha component is ignored, and the results
       are as if alpha was 1.0.

New Procedures and Functions

    None

New Tokens

    Accepted by the <pname> parameter of TexParameter* and GetTexParameter*:

        TEXTURE_ASTC_DECODE_PRECISION_EXT   0x8F69

Changes to 8.10 Texture Parameters

   Add to Table 8.19: Texture parameters and their values:

   -----------------------------------------------------------------------------
   Name                               Type                Legal Values
   -----------------------------------------------------------------------------
   TEXTURE_ASTC_DECODE_PRECISION_EXT  enum                GL_RGBA16F
                                                          GL_RGBA8
                                                          GL_RGB9_E5 (*)
   -----------------------------------------------------------------------------

   (*) Note: GL_RGB9_E5 is legal if and only if the
   GL_EXT_texture_compression_astc_decode_mode_rgb9e5 extension is supported.

Changes to C.2.5 LDR and HDR Modes

    In Table C.2.1 (Differences Between LDR and HDR Modes) in KHR_texture_compression_astc_hdr:

    Modify the first row of the table to read:

    -----------------------------------------------------------------------------
    Operation           LDR Mode                    HDR Mode
    -----------------------------------------------------------------------------
    Returned value      Determined by the           Determined by the
                        decoding mode               decoding mode

    <the rest of the table remains unchanged>
    -----------------------------------------------------------------------------
          Table C.2.1 - Differences Between LDR and HDR Modes


    Add the following paragraph immediately after Table C.2.1:

    The type of the values returned by the decoding process is determined by API
    state as follows:

    -----------------------------------------------------------------------------
    Decode mode              LDR Mode                     HDR Mode
    -----------------------------------------------------------------------------
    GL_RGBA16F               Vector of FP16 values        Vector of FP16 values
    GL_RGBA8                 Vector of UNORM8 values      <invalid>
    GL_RGB9_E5               Vector using a shared        Vector using a shared
                             exponent format              exponent format
    -----------------------------------------------------------------------------
          Table C.2.1.a - Decoding mode

    [[ HDR profile only ]]
    Using the GL_RGBA8 decoding mode in HDR mode gives undefined results.

    For sRGB output, the decoding mode is ignored, and the decoding always returns
    a vector of UNORM8 values.

    If the texture does not have an ASTC format, the decoding mode is ignored.


    [[ If GL_EXT_texture_compression_astc_decode_mode_rgb9e5 and HDR profile
       is supported, then add the following paragraph after the second paragraph
       following Table C.2.1 ]]

    When using the GL_RGB9_E5 decoding mode in HDR mode, error results
    will return the error color because NaN cannot be represented.


Changes to C.2.19 (Weight Application)

    Replace the paragraph immediately after the expression for the value C in
    LDR mode with the following:

    If sRGB conversion is not enabled and the decoding mode is GL_RGBA16F,
    then if C = 65535, then the final result is 1.0 (0x3C00) otherwise C is divided by
    65536 and the infinite-precision result of the division is converted to FP16 with
    round-to-zero semantics.

    If sRGB conversion is not enabled and the decoding mode is GL_RGBA8,
    then top 8 bits of the interpolation result for the R, G, B, and A channels
    are used as the final result.

   [[ The following two paragraph applies to
      GL_EXT_texture_compression_astc_decode_mode_rgb9e5 only. ]]

    If sRGB conversion is not enabled and the decoding mode is GL_RGB9_E5,
    then the final result is a combination of the (UNORM16) values of C for the three
    color components (Cr, Cg, and Cb) computed as follows:

        int lz = clz17( Cr | Cg | Cb | 1);
        if (Cr == 65535 ) { Cr = 65536; lz = 0; }
        if (Cg == 65535 ) { Cg = 65536; lz = 0; }
        if (Cb == 65535 ) { Cb = 65536; lz = 0; }
        Cr <<= lz;
        Cg <<= lz;
        Cb <<= lz;
        Cr = (Cr >> 8) & 0x1FF;
        Cg = (Cg >> 8) & 0x1FF;
        Cb = (Cb >> 8) & 0x1FF;
        uint32_t exponent = 16 - lz;

        uint32_t texel = (exponent << 27) | (Cb << 18) | (Cg << 9) | Cr;

    The clz17() function counts leading zeros in a 17-bit value.


    If sRGB conversion is enabled, then the decoding mode is ignored,
    and the top 8 bits of the interpolation result for the R, G and B
    channels are passed to the external sRGB conversion block and used
    as the final result.


    [[ If GL_EXT_texture_compression_astc_decode_mode_rgb9e5 and HDR profile
       is supported, then add the following at the end of the section. ]]

    If the decoding mode is GL_RGB9_E5, then the final result
    is a combination of the (IEEE FP16) values of Cf for the three color
    components (Cr, Cg, and Cb) computed as follows:

        if( Cr > 0x7c00 ) Cr = 0; else if( Cr == 0x7c00 ) Cr = 0x7bff;
        if( Cg > 0x7c00 ) Cg = 0; else if( Cg == 0x7c00 ) Cg = 0x7bff;
        if( Cb > 0x7c00 ) Cb = 0; else if( Cb == 0x7c00 ) Cb = 0x7bff;
        int Re = (Cr >> 10) & 0x1F;
        int Ge = (Cg >> 10) & 0x1F;
        int Be = (Cb >> 10) & 0x1F;
        int Rex = Re == 0 ? 1 : Re;
        int Gex = Ge == 0 ? 1 : Ge;
        int Bex = Be == 0 ? 1 : Be;
        int Xm = ((Cr | Cg | Cb) & 0x200) >> 9;
        int Xe = Re | Ge | Be;
        uint32_t rshift, gshift, bshift, expo;

        if (Xe == 0)
        {
            expo = rshift = gshift = bshift = Xm;
        }
        else if (Re >= Ge && Re >= Be)
        {
            expo = Rex + 1;
            rshift = 2;
            gshift = Rex - Gex + 2;
            bshift = Rex - Bex + 2;
        }
        else if (Ge >= Be)
        {
            expo = Gex + 1;
            rshift = Gex - Rex + 2;
            gshift = 2;
            bshift = Gex - Bex + 2;
        }
        else
        {
            expo = Bex + 1;
            rshift = Bex - Rex + 2;
            gshift = Bex - Gex + 2;
            bshift = 2;
        }

        int Rm = (Cr & 0x3FF) | (Re == 0 ? 0 : 0x400);
        int Gm = (Cg & 0x3FF) | (Ge == 0 ? 0 : 0x400);
        int Bm = (Cb & 0x3FF) | (Be == 0 ? 0 : 0x400);
        Rm = (Rm >> rshift) & 0x1FF;
        Gm = (Gm >> gshift) & 0x1FF;
        Bm = (Bm >> bshift) & 0x1FF;

        uint32_t texel = (expo << 27) | (Bm << 18) | (Gm << 9) | (Rm << 0);

Changes to C.2.23  (Void-Extent Blocks)

    [[ If GL_EXT_texture_compression_astc_decode_mode_rgb9e5 and HDR profile
       is supported, then add the following paragraph. ]]

    In the HDR case, if the decoding mode is GL_RGB9_E5, then
    any negative color component values are set to 0 before conversion to
    the shared exponent format (as described in C.2.19).

New State

    In Table 21.10: Textures (state per texture object), add:

    Get value               Type Get Command     Initial value  Description Section
    ----                    ---- -------------   -------------  ----------- -------
    TEXTURE_ASTC_DECODE_-   E    GetTexParameter GL_RGBA16F     Decode mode C.2.5
    PRECISION_EXT


Interactions with GL_KHR_texture_compression_astc_hdr

    If GL_KHR_texture_compression_astc_hdr is not supported, then the
    HDR profile is not supported, all references to HDR are removed.

Issues

    (1) Should we include the GL_RGB9_E5 mode?

        Proposed: Yes.

    (2) Are implementations allowed to decode at a higher precision than
        what is requested?

        Proposed: No.

        If we allow this, then this extension could be exposed on all
        implementations that support ASTC. But developers would have no
        way of knowing what precision was actually used, and thus whether
        image quality is sufficient at reduced precision.

    (3) What happens to values in void-extent blocks that are infinity or
        NaN when using GL_RGB9_E5?

        Resolved: Undefined behavior.

        KHR_texture_compression_astc_hdr already makes this undefined:

        "In the HDR case, if the color component values are infinity or NaN,
         this will result in undefined behavior. As usual, this must not lead
         to GL interruption or termination."

    (4) Should there be an error or a texture completeness rule when using
        the GL_RGBA8 decoding mode for HDR mode blocks?

        Resolved: No. There is no way for implementations to check this
        condition short of iterating through all the blocks of the texture.

    (5) Should the decode mode be texture image state and/or sampler state?

        Resolved: Texture image state only. Some implementations effectively
        treat the the different decode modes as different texture formats.

Revision History

    Revision 4, January 23, 2017  - Split functionality in two name strings.
                                    Clarified interaction with
                                    KHR_texture_compression_astc_hdr
                                    Added issue 5.
    Revision 3, January 18, 2017  - Tidy up.
    Revision 2, December 6, 2016  - Fixed type errors in rounding operation.
                                    Clarified behavior of negative values in
                                    void-extent blocks when decoding to shared
                                    exponent format.
    Revision 1, November 11, 2016 - Initial draft.

