# NV_conservative_raster

Name

    NV_conservative_raster

Name Strings

    GL_NV_conservative_raster

Contact

    Jeff Bolz, NVIDIA Corporation (jbolz 'at' nvidia.com)

Contributors

    Michael Chock, NVIDIA Corporation

Status

    Shipping.

Version

    Last Modified Date:         March 27, 2015
    Revision:                   3

Number

    OpenGL Extension #465
    OpenGL ES Extension #228

Dependencies

    This extension is written against the OpenGL 4.3 specification
    (Compatibility Profile) but may be used with the Core profile or
    OpenGL ES 2.0 or later.

    When this extension is used with the Core profile or an OpenGL ES
    context, references to functionality specific to the Compatibility
    Profile can be ignored.
    
Overview

    This extension adds a "conservative" rasterization mode where any pixel
    that is partially covered, even if no sample location is covered, is 
    treated as fully covered and a corresponding fragment will be shaded.

    A new control is also added to modify window coordinate snapping 
    precision.

    These controls can be used to implement "binning" to a low-resolution
    render target, for example to determine which tiles of a sparse texture
    need to be populated. An app can construct a framebuffer where there is
    one pixel per tile in the sparse texture, and adjust the number of
    subpixel bits such that snapping occurs to the same effective grid as when
    rendering to the sparse texture. Then triangles should cover (at least)
    the same pixels in the low-res framebuffer as they do tiles in the sparse
    texture.


New Procedures and Functions

    void SubpixelPrecisionBiasNV(uint xbits, uint ybits);

New Tokens

    Accepted by the <cap> parameter of Enable, Disable, IsEnabled:

        CONSERVATIVE_RASTERIZATION_NV                   0x9346

    Accepted by the <pname> parameter of GetBooleanv, GetDoublev,
    GetIntegerv, and GetFloatv:

        SUBPIXEL_PRECISION_BIAS_X_BITS_NV               0x9347
        SUBPIXEL_PRECISION_BIAS_Y_BITS_NV               0x9348
        MAX_SUBPIXEL_PRECISION_BIAS_BITS_NV             0x9349
    
Additions to Chapter 13 of the OpenGL 4.3 (Compatibility Profile) Specification
(Fixed-Function Vertex Post-Processing)

    Modify subsection 13.6.1 "Controlling the Viewport", p. 469

    Add after the formula for the vertex's window coordinates:

    The vertex's window x and y coordinates may be optionally converted 
    to fixed-point values with <N> fractional bits. If CONSERVATIVE_-
    RASTERIZATION_NV is disabled (see section 14.6.X), then <N> is the 
    implementation-dependent value of SUBPIXEL_BITS. If CONSERVATIVE_-
    RASTERIZATION_NV is enabled, <N> is computed as the sum of the value of 
    SUBPIXEL_BITS and programmable values of SUBPIXEL_PRECISION_BIAS_{X,Y}-
    _BITS_NV. These values may be set with the command

        void SubpixelPrecisionBiasNV(uint xbits, uint ybits);

    When these values are non-zero, the invariance requirement of section 14.2 
    may not apply because the subpixel precision may not be the same at all 
    window coordinates. The initial values of SUBPIXEL_PRECISION_BIAS_{X,Y}-
    _BITS_NV are zero. If <xbits> or <ybits> are greater than the value of 
    MAX_SUBPIXEL_PRECISION_BIAS_BITS_NV, the error INVALID_VALUE is generated.


Additions to Chapter 14 of the OpenGL 4.3 (Compatibility Profile) Specification
(Fixed-Function Primitive Assembly and Rasterization)

    Add a new subsection at the end of 14.6, 14.6.X "Conservative Rasterization"

    Point, line, and polygon rasterization may optionally be made conservative
    by calling Enable and Disable with a <pname> of CONSERVATIVE_-
    RASTERIZATION_NV. When conservative rasterization is enabled, rather than 
    evaluating coverage at individual sample locations, a determination is made 
    of whether any portion of the pixel (including its edges and corners) is 
    covered by the primitive. If any portion of the pixel is covered, then a 
    fragment is generated with all coverage samples turned on. Conservative 
    rasterization may also generate fragments for pixels near the edges of 
    rasterized point or line primitives, even if those pixels are not covered 
    by the primitive. The set of such pixels is implementation-dependent, but 
    implementations are encouraged to evaluate coverage as precisely as 
    possible.

    If CONSERVATIVE_RASTERIZATION_NV is enabled, points are rasterized 
    according to multisample rasterization rules (section 14.4.3), except that 
    a fragment will be generated for a framebuffer pixel if the circle 
    (POINT_SPRITE disabled) or square (POINT_SPRITE enabled) covers any portion 
    of the pixel, including its edges or corners.  When performing conservative 
    rasterization of points, the POINT_SMOOTH enable is ignored and treated as
    disabled.

    If CONSERVATIVE_RASTERIZATION_NV is enabled, lines are rasterized according 
    to multisample rasterization rules (section 14.5.4), except that the 
    LINE_STIPPLE and LINE_SMOOTH enables are ignored and treated as disabled.

    If CONSERVATIVE_RASTERIZATION_NV is enabled, polygons are rasterized 
    according to multisample rasterization rules (section 14.6.6), except that 
    the POLYGON_SMOOTH enable is ignored and treated as disabled. Polygons with 
    an area of zero generate no fragments, even for pixels that contain a 
    vertex or edge of the zero-area polygon.

    Modify the new Subsection "Drawing Textures" from the NV_draw_texture 
    extension:

    In either case, the set of fragments generated is not affected by the 
    CULL_FACE, POLYGON_SMOOTH, POLYGON_OFFSET_FILL enables, or PolygonMode 
    state. The CONVSERVATIVE_RASTERIZATION_NV enable does apply, and fragments
    will be generated for all pixels which have any portion covered by the 
    rectangle. All fragments generated for the rectangle will have a Z window
    coordinate of <z>.


Interactions with OpenGL ES and Core Profiles

    If using OpenGL ES or a Core profile, references to LINE_SMOOTH,
    LINE_STIPPLE, POINT_SMOOTH, and POLYGON_SMOOTH are ignored and treated as
    disabled. POINT_SPRITE is ignored and treated as enabled. For OpenGL ES,
    references to PolygonMode are ignored.


New Implementation Dependent State

                                                      Minimum
    Get Value                    Type    Get Command  Value   Description                   Sec.
    ---------                    ------- -----------  ------- ------------------------      ------
    MAX_SUBPIXEL_PRECISION_-     Z+      GetIntegerv   1      Max number of extra bits      13.6.1
        BIAS_BITS_NV


New State

    Get Value                           Get Command    Type    Initial Value    Description                 Sec.    Attribute
    ---------                           -----------    ----    -------------    -----------                 ----    ---------
    CONSERVATIVE_RASTERIZATION_NV       IsEnabled      B       FALSE            Enable conservative         14.6.X  enable
                                                                                rasterization rules
    SUBPIXEL_PRECISION_BIAS_X_BITS_NV   GetIntegerv    Z+      0                Additional window x         13.6.1  viewport
                                                                                coordinate precision
    SUBPIXEL_PRECISION_BIAS_Y_BITS_NV   GetIntegerv    Z+      0                Additional window y         13.6.1  viewport
                                                                                coordinate precision

Additions to the AGL/GLX/WGL Specifications

    None.

GLX Protocol

    None.
    
Modifications to the OpenGL Shading Language Specification, Version 4.30

    None.

Errors

    INVALID_VALUE is generated by SubpixelPrecisionBiasNV if <xbits> or
    <ybits> are greater than the value of MAX_SUBPIXEL_PRECISION_BIAS_BITS_NV.

Issues

    (1) How is invariance affected by the precision bias?

    RESOLVED: Invariance may be broken for large enough values of the bias. It 
    is expected that an implementation has enough precision to support 
    SUBPIXEL_BITS for a MAX_VIEWPORT_DIMS size viewport, but if the combination 
    of viewport size and total subpixel precision exceed that then less 
    precision may be used for large x,y coordinates.

    (2) Do zero area primitives generate fragments in conservative raster?

    RESOLVED: No, although in some cases that may not be the desired behavior.
    If a primitive is truly zero area (e.g. two vertices of a triangle have 
    identical positions), then drawing nothing is probably fine. If the 
    primitive happens to be zero area due to subpixel precision then generating
    fragments may be desirable, but this spec does define that behavior.

    The primary reason to discard zero area primitives is that attribute 
    interpolation is not well-defined when the area is zero.

    (3) How does centroid interpolation work for a conservative primitive?

    RESOLVED: Since a fragment generated by a conservative primitive is 
    considered "fully covered", any location within the pixel may be used for
    interpolation. This implies that the interpolation may occur outside of the
    original primitive, causing attribute extrapolation. 
    
    (4) How is depth coordinate evaluation handled for conservative 
    rasterization?
    
    RESOLVED: The "extrapolation" issue for attributes also applies to depth
    evaluation.

    (5) How does NV_draw_texture interact with this extension?

    RESOLVED: The DrawTextureNV command rasterizes conservatively.

Revision History

    Revision 3, 2015/03/27
      - Add ES interactions

    Revision 2, 2014/09/26 (Jon Leech)
      - Add missing return type to function 

    Revision 1
      - Internal revisions.
