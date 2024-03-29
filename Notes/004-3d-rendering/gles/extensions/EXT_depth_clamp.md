# EXT_depth_clamp

Name

    EXT_depth_clamp

Name Strings

    GL_EXT_depth_clamp

Contact

    Gert Wollny (gert.wollny 'at' collabora.com)

Notice

    Copyright (c) 2019 Collabora LTD 
    Copyright (c) 2009-2013 The Khronos Group Inc. Copyright terms at
        http://www.khronos.org/registry/speccopyright.html

Status

    Complete.

Version

    Version 1, 2019/01/24.
    Based on ARB_depth_clamp version 4, modified 2009/08/02.

Number

    #309

Dependencies

    OpenGL ES 2.0 is required.
    Written based on the wording of the OpenGL ES 3.2 specification.

Overview

    Conventional OpenGL clips geometric primitives to a clip volume
    with six faces, two of which are the near and far clip planes.
    Clipping to the near and far planes of the clip volume ensures that
    interpolated depth values (after the depth range transform) must be
    in the [0,1] range.

    In some rendering applications such as shadow volumes, it is useful
    to allow line and polygon primitives to be rasterized without
    clipping the primitive to the near or far clip volume planes (side
    clip volume planes clip normally).  Without the near and far clip
    planes, rasterization (pixel coverage determination) in X and Y
    can proceed normally if we ignore the near and far clip planes.
    The one major issue is that fragments of a  primitive may extend
    beyond the conventional window space depth range for depth values
    (typically the range [0,1]).  Rather than discarding fragments that
    defy the window space depth range (effectively what near and far
    plane clipping accomplish), the depth values can be clamped to the
    current depth range.

    This extension provides exactly such functionality.  This
    functionality is useful to obviate the need for near plane capping
    of stenciled shadow volumes.  The functionality may also be useful
    for rendering geometry "beyond" the far plane if an alternative
    algorithm (rather than depth testing) for hidden surface removal is
    applied to such geometry (specifically, the painter's algorithm).
    Similar situations at the near clip plane can be avoided at the
    near clip plane where apparently solid objects can be "seen through"
    if they intersect the near clip plane.

New Procedures and Functions

    None

New Tokens

    Accepted by the <cap> parameter of Enable, Disable, and IsEnabled,
    and by the <pname> parameter of GetBooleanv, GetIntegerv,
    GetFloatv:

        DEPTH_CLAMP_EXT                               0x864F

Additions to Chapter 12 of the OpenGL ES 3.2 Specification (Fixed-Function Vertex Post-Processing)

 --  Section 12.5 "Primitive Clipping"

    Add to the end of the 1st paragraph:

    "Depth clamping is enabled with the generic Enable command and
    disabled with the Disable command. The value of the argument to
    either command is DEPTH_CLAMP_EXT. If depth clamping is enabled, the
    "-w_c <= z_c <= w_c" plane equation are ignored by view volume
    clipping (effectively, there is no near or far plane clipping)."

Additions to Chapter 15 of the OpenGL ES 3.2 Specification (Writing Fragments and Samples to the Framebuffer)

 --  Section 15.1.3 "Depth buffer test"

    Add to the end of the 2nd paragraph:

    "If depth clamping (see section 12.15) is enabled, before the
    incoming fragment's z_w is compared z_w is clamped to the range
    [min(n,f),max(n,f)], where n and f are the current near and far
    depth range values (see section 12.6.1)."

Additions to the AGL/GLX/WGL Specifications

    None

GLX Protocol

    None

Errors

    None

New State

Add to table 6.4, transformation state

Get Value       Type  Get Command  Initial Value  Description     Sec    
--------------  ----  -----------  -------------  --------------  ------ 
DEPTH_CLAMP_EXT  B     IsEnabled    False          Depth clamping  12.5  
                                                  on/off

New Implementation Dependent State

    None

Issues

    See the issue list in GL_ARB_depth_clamp.

    Can fragments with w_c <=0 be generated when this extension is supported?
 
      RESOLUTION: No. The inequalities in OpenGL ES Specification 12.5 clarify 
      that only primitives that lie in the region w_c >= 0 can be produced by
      clipping and the vertex normalization in 12.6 clarifies that values 
      w_c = 0 are prohibited. Hence fragments with w_c <= 0 should also never
      be generated when this extension is supported.

    How does this extension differ from ARB_depth_clamp? 
      
      Instead of DEPTH_CLAMP the parameter is called DEPTH_CLAMP_EXT.
      Push/pop attrib bits are not relevant for OpenGL ES. 

Revision History

    Version 1, 2019/01/25 (Gert Wollny) - rewrite ARB_depth_clamp against
    OpenGL ES 3.2 instead of OpenGL 3.1.
