# 


Name
    
    EXT_texture_filter_anisotropic

Name Strings

    GL_EXT_texture_filter_anisotropic

Notice

    Copyright NVIDIA Corporation, 1999.

Version

    Last updated May 23, 2018

Number

    OpenGL Extension #187
    OpenGL ES Extension #41

Dependencies

    Written based on the wording of the OpenGL 1.2 specification.

Overview

    Texture mapping using OpenGL's existing mipmap texture filtering
    modes assumes that the projection of the pixel filter footprint into
    texture space is a square (ie, isotropic).  In practice however, the
    footprint may be long and narrow (ie, anisotropic).  Consequently,
    mipmap filtering severely blurs images on surfaces angled obliquely
    away from the viewer.

    Several approaches exist for improving texture sampling by accounting
    for the anisotropic nature of the pixel filter footprint into texture
    space.  This extension provides a general mechanism for supporting
    anisotropic texturing filtering schemes without specifying a
    particular formulation of anisotropic filtering.

    The extension permits the OpenGL application to specify on
    a per-texture object basis the maximum degree of anisotropy to
    account for in texture filtering.

    Increasing a texture object's maximum degree of anisotropy may
    improve texture filtering but may also significantly reduce the
    implementation's texture filtering rate.  Implementations are free
    to clamp the specified degree of anisotropy to the implementation's
    maximum supported degree of anisotropy.

    A texture's maximum degree of anisotropy is specified independent
    from the texture's minification and magnification filter (as
    opposed to being supported as an entirely new filtering mode).
    Implementations are free to use the specified minification and
    magnification filter to select a particular anisotropic texture
    filtering scheme.  For example, a NEAREST filter with a maximum
    degree of anisotropy of two could be treated as a 2-tap filter that
    accounts for the direction of anisotropy.  Implementations are also
    permitted to ignore the minification or magnification filter and
    implement the highest quality of anisotropic filtering possible.

    Applications seeking the highest quality anisotropic filtering
    available are advised to request a LINEAR_MIPMAP_LINEAR minification
    filter, a LINEAR magnification filter, and a large maximum degree
    of anisotropy.

Issues

    Should there be a particular anisotropic texture filtering minification
    and magnification mode?

      RESOLUTION:  NO.  The maximum degree of anisotropy should control
      when anisotropic texturing is used.  Making this orthogonal to
      the minification and magnification filtering modes allows these
      settings to influence the anisotropic scheme used.  Yes, such
      an anisotropic filtering scheme exists in hardware.

    What should the minimum value for MAX_TEXTURE_MAX_ANISOTROPY_EXT be?

      RESOLUTION:  2.0.  To support this extension, at least 2 to 1
      anisotropy should be supported.

    Should an implementation-defined limit for the maximum maximum degree of
    anisotropy be "get-able"?

      RESOLUTION:  YES.  But you should not assume that a high maximum
      maximum degree of anisotropy implies anything about texture
      filtering performance or quality.

    Should anything particular be said about anisotropic 3D texture filtering?

      Not sure.  Does the implementation example shown in the spec for
      2D anisotropic texture filtering readily extend to 3D anisotropic
      texture filtering?

New Procedures and Functions

    None

New Tokens

    Accepted by the <pname> parameters of GetTexParameterfv,
    GetTexParameteriv, TexParameterf, TexParameterfv, TexParameteri,
    and TexParameteriv:

        TEXTURE_MAX_ANISOTROPY_EXT          0x84FE

    Accepted by the <pname> parameters of GetBooleanv, GetDoublev,
    GetFloatv, and GetIntegerv:

        MAX_TEXTURE_MAX_ANISOTROPY_EXT      0x84FF

Additions to Chapter 2 of the 1.2 Specification (OpenGL Operation)

     None

Additions to Chapter 3 of the 1.2 Specification (Rasterization)

 --  Sections 3.8.3 "Texture Parameters"

     Add the following entry to the end of Table 3.17:

     Name                         Type    Legal Values
     --------------------------   ------  --------------------------
     TEXTURE_MAX_ANISOTROPY_EXT   float   greater or equal to 1.0


 --  Sections 3.8.5 "Texture Minification" and 3.8.6 "Texture Magnification"

     After the first paragraph in Section 3.8.5:
 
     "When the texture's value of TEXTURE_MAX_ANISOTROPY_EXT is equal to 1.0,
     the GL uses an isotropic texture filtering approach as described in
     this section and Section 3.8.6.  However, when the texture's value
     of TEXTURE_MAX_ANISOTROPY_EXT is greater than 1.0, the GL implementation
     should use a texture filtering scheme that accounts for a degree
     of anisotropy up to the smaller of the texture's value of
     TEXTURE_MAX_ANISOTROPY_EXT or the implementation-defined value of
     MAX_TEXTURE_MAX_ANISOTROPY_EXT.

     The particular scheme for anisotropic texture filtering is
     implementation dependent.  Additionally, implementations are free
     to consider the current texture minification and magnification modes
     to control the specifics of the anisotropic filtering scheme used.

     The anisotropic texture filtering scheme may only access mipmap
     levels if the minification filter is one that requires mipmaps.
     Additionally, when a minification filter is specified, the
     anisotropic texture filtering scheme may only access texture mipmap
     levels between the texture's values for TEXTURE_BASE_LEVEL and
     TEXTURE_MAX_LEVEL, inclusive.  Implementations are also recommended
     to respect the values of TEXTURE_MAX_LOD and TEXTURE_MIN_LOD to
     whatever extent the particular anisotropic texture filtering
     scheme permits this."

     The following describes one particular approach to implementing
     anisotropic texture filtering for the 2D texturing case:

     "Anisotropic texture filtering substantially changes Section 3.8.5.
     Previously a single scale factor P was determined based on the
     pixel's projection into texture space.  Now two scale factors,
     Px and Py, are computed.

       Px = sqrt(dudx^2 + dvdx^2)
       Py = sqrt(dudy^2 + dvdy^2)

       Pmax = max(Px,Py)
       Pmin = min(Px,Py)

       N = min(ceil(Pmax/Pmin),maxAniso)
       Lamda' = log2(Pmax/N)

     where maxAniso is the smaller of the texture's value of
     TEXTURE_MAX_ANISOTROPY_EXT or the implementation-defined value of
     MAX_TEXTURE_MAX_ANISOTROPY_EXT.

     It is acceptable for implementation to round 'N' up to the nearest
     supported sampling rate.  For example an implementation may only
     support power-of-two sampling rates.

     It is also acceptable for an implementation to approximate the ideal
     functions Px and Py with functions Fx and Fy subject to the following
     conditions:

       1.  Fx is continuous and monotonically increasing in |du/dx| and |dv/dx|.
           Fy is continuous and monotonically increasing in |du/dy| and |dv/dy|.

       2.  max(|du/dx|,|dv/dx|} <= Fx <= |du/dx| + |dv/dx|.
           max(|du/dy|,|dv/dy|} <= Fy <= |du/dy| + |dv/dy|.

     Instead of a single sample, Tau, at (u,v,Lamda), 'N' locations in the mipmap
     at LOD Lamda, are sampled within the texture footprint of the pixel.

     Instead of a single sample, Tau, at (u,v,lambda), 'N' locations in
     the mipmap at LOD Lamda are sampled within the texture footprint of
     the pixel.  This sum TauAniso is defined using the single sample Tau.
     When the texture's value of TEXTURE_MAX_ANISOTROPHY_EXT is greater
     than 1.0, use TauAniso instead of Tau to determine the fragment's
     texture value.

                    i=N
                    ---
     TauAniso = 1/N \ Tau(u(x - 1/2 + i/(N+1), y), v(x - 1/2 + i/(N+1), y)),  Px > Py
                    /
                    ---
                    i=1

                    i=N
                    ---
     TauAniso = 1/N \ Tau(u(x, y - 1/2 + i/(N+1)), v(x, y - 1/2 + i/(N+1))),  Py >= Px
                    /
                    ---
                    i=1


     It is acceptable to approximate the u and v functions with equally spaced
     samples in texture space at LOD Lamda:

                    i=N
                    ---
     TauAniso = 1/N \ Tau(u(x,y)+dudx(i/(N+1)-1/2), v(x,y)+dvdx(i/(N+1)-1/2)), Px > Py
                    /
                    ---
                    i=1

                    i=N
                    ---
     TauAniso = 1/N \ Tau(u(x,y)+dudy(i/(N+1)-1/2), v(x,y)+dvdy(i/(N+1)-1/2)), Py >= Px
                    /
                    ---
                    i=1 

     "

Additions to Chapter 4 of the 1.2 Specification (Per-Fragment Operations
and the Frame Buffer)

     None

Additions to Chapter 5 of the 1.2 Specification (Special Functions)

     None

Additions to Chapter 6 of the 1.2 Specification (State and State Requests)

     None

Additions to the GLX Specification

     None

Errors

     INVALID_VALUE is generated when TexParameter is called with <pname>
     of TEXTURE_MAX_ANISOTROPY_EXT and a <param> value or value of what
     <params> points to less than 1.0.

New State

(table 6.13, p203) add the entry:

Get Value                   Type  Get Command        Initial Value   Description      Sec     Attribute
--------------------------  ----  -----------------  --------------  ---------------  -----   ---------
TEXTURE_MAX_ANISOTROPY_EXT   R    GetTexParameterfv  1.0             Maximum degree   3.8.5    texture
                                                                     of anisotropy

New Implementation State

(table 6.25, p215) add the entry:

Get Value                       Type  Get Command   Minimum Value   Description      Sec     Attribute
------------------------------  ----  ------------  --------------  ---------------  -----   ---------
MAX_TEXTURE_MAX_ANISOTROPY_EXT   R    GetFloatv     2.0             Limit of         3.8.5    -
                                                                    maximum degree
                                                                    of anisotropy

Issues

  1) Should TEXTURE_MAX_ANISOTROPY_EXT be accepted by SamplerParameter*?

  Yes, for implementations supporting sampler objects. The per-texture sampling
  state is overridden by the sampler object state, if present. The anisotropy
  parameter should not be an exception, as this would reduce the usefulness of
  sampler objects when anisotropic filtering is supported. This also matches
  the interaction described in ARB_sampler_objects, and the same behavior is
  still expected for API versions with core support for sampler objects.

Revision History

  2018-05-23 (Nicolas Capens) - clarify interaction with sampler objects.

  11/12/14 (Jon Leech) - Fix spelling of TEXTURE_MAX_ANISOTROPY 
  (public Bug 1263).

  9/26/07 (Jon Leech) - assigned OpenGL ES extension number so
  the extension can live in both API registries.

  4/25/00 - clarify that TexParameterf and TexParameteri accept
  TEXTURE_MAX_ANISOTROPY_EXT as a pname.

