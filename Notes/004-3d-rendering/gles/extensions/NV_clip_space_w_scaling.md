# NV_clip_space_w_scaling

Name

    NV_clip_space_w_scaling

Name Strings

    GL_NV_clip_space_w_scaling

Contact

    Kedarnath Thangudu, NVIDIA Corporation (kthangudu 'at' nvidia.com)

Contributors

    Eric Werness, NVIDIA Corporation
    Ingo Esser, NVIDIA Corporation
    Pat Brown, NVIDIA Corporation
    Mark Kilgard, NVIDIA Corporation
    Jason Schmidt, NVIDIA Corporation

Status

    Shipping in NVIDIA release 367.XX drivers and up.

Version

    Last Modified Date:         November 25, 2017
    Revision:                   4

Number

    OpenGL Extension #486
    OpenGL ES Extension #295

Dependencies

    This extension is written against OpenGL 4.5 Specification
    (Compatibility Profile).

    This extension interacts with the OpenGL ES 3.1 Specification.

    This extension requires NV_viewport_array2.

    If implemented in OpenGL ES, one of NV_viewport_array or OES_viewport_array
    is required.

Overview

    Virtual Reality (VR) applications often involve a post-processing step to
    apply a "barrel" distortion to the rendered image to correct the
    "pincushion" distortion introduced by the optics in a VR device. The
    barrel distorted image has lower resolution along the edges compared to
    the center.  Since the original image is rendered at high resolution,
    which is uniform across the complete image, a lot of pixels towards the
    edges do not make it to the final post-processed image.

    This extension also provides a mechanism to render VR scenes at a
    non-uniform resolution, in particular a resolution that falls linearly
    from the center towards the edges.  This is achieved by scaling the "w"
    coordinate of the vertices in the clip space before perspective divide.
    The clip space "w" coordinate of the vertices may be offset as of a
    function of "x" and "y" coordinates as follows:

            w' = w + Ax + By

    In the intended use case for viewport position scaling, an application
    should use a set of 4 viewports, one for each of the 4 quadrants of a
    Cartesian coordinate system.  Each viewport is set to the dimension of the
    image, but is scissored to the quadrant it represents.  The application
    should specify A and B coefficients of the w-scaling equation above,
    that have the same value, but different signs, for each of the viewports.
    The signs of A and B should match the signs of X and Y for the quadrant
    that they represent such that the value of "w'" will always be greater
    than or equal to the original "w" value for the entire image. Since the
    offset to "w", (Ax + By), is always positive and increases with the
    absolute values of "x" and "y", the effective resolution will fall off
    linearly from the center of the image to its edges.

New Procedures and Functions

    void ViewportPositionWScaleNV(uint index, float xcoeff, float ycoeff)

New Tokens

    Accepted by the <cap> parameter of Enable, Disable, IsEnabled:

        VIEWPORT_POSITION_W_SCALE_NV            0x937C

    Accepted by the <pname> parameter of GetBooleani_v, GetDoublei_v,
    GetIntegeri_v, GetFloati_v, and GetInteger64i_v:

        VIEWPORT_POSITION_W_SCALE_X_COEFF_NV    0x937D
        VIEWPORT_POSITION_W_SCALE_Y_COEFF_NV    0x937E

Additions to Chapter 13 of the OpenGL 4.5 (Compatibility Profile)
Specification (Fixed-Function Vertex Post-Processing)

    Modify Section 13.2 (Transform Feedback), p. 453 [section 12.1 in OpenGL ES]

    Modify the first paragraph:

    ...The vertices are fed back after vertex color clamping, but before
    viewport mask expansion, w coordinate warping, flat-shading, and clipping...

    Add a new Section 13.X (Viewport W Coordinate Scaling)

    If VIEWPORT_POSITION_W_SCALE_NV is enabled, the w coordinates for each
    primitive sent to a given viewport will be scaled as a function of
    its x and y coordinates using the following equation:

        w' = xcoeff * x + ycoeff * y + w;

    The coefficients for "x" and "y" used in the above equation depend on the
    viewport index, and are controlled by the command

        void ViewportPositionWScaleNV(uint index, float xcoeff, float ycoeff);

    The viewport specified by <index> has its coefficients for "x" and "y"
    set to the <xcoeff> and <ycoeff> values.  Specifying these coefficients
    enables rendering images at a non-uniform resolution, in particular a
    resolution that falls off linearly from the center towards the edges,
    which is useful for VR applications. VR applications often involve a
    post-processing step to apply a "barrel" distortion to the rendered image
    to correct the "pincushion" distortion introduced by the optics in a VR
    device. The barrel distorted image, has lower resolution along the edges
    compared to the center.  Since the original image is rendered at high
    resolution, which is uniform across the complete image, a lot of pixels
    towards the edges do not make it to the final post-processed image.
    VR applications may use the w-scaling to minimize the processing of unused
    fragments. To achieve the intended effect, applications should use a set of
    4 viewports one for each of the 4 quadrants of a Cartesian coordinate
    system.  Each viewport is set to the dimension of the image, but is
    scissored to the quadrant it represents.  The application should specify
    the x and y coefficients of the w-scaling equation above, that have the
    same value, but different signs, for each of the viewports.  The signs of
    <xcoeff> and <ycoeff> should match the signs of X and Y for the quadrant
    that they represent such that the value of "w'" will always be greater
    than or equal to the original "w" value for the entire image. Since the
    offset to "w", (Ax + By), is always positive and increases with the
    absolute values of "x" and "y", the effective resolution will fall off
    linearly from the center of the image to its edges.

    Errors:

    - The error INVALID_VALUE is generated if <index> is greater than or equal
      to the value of MAX_VIEWPORTS.

New Implementation Dependent State

    None.

New State

                                                                 Initial
    Get Value                             Get Command    Type    Value     Description                  Sec.    Attribute
    ------------------------------------  -----------    ----    -------   -----------                  ----    ---------
    VIEWPORT_POSITION_W_SCALE_NV          IsEnabled      B       FALSE     Enable W coordinate Scaling  13.X    enable
    VIEWPORT_POSITION_W_SCALE_X_COEFF_NV  GetFloati_v    R       0         x coefficient for the w      13.X    viewport
                                                                           coordinate scaling equation
    VIEWPORT_POSITION_W_SCALE_Y_COEFF_NV  GetFloati_v    R       0         y coefficient for the w      13.X    viewport
                                                                           coordinate scaling equation

Additions to the AGL/GLX/WGL/EGL Specifications

    None.

GLX Protocol

    None.

Errors

    None.

Interactions with OpenGL ES 3.1

    If implemented in OpenGL ES, remove all references to GetDoublei_v.
    If NV_viewport_array is supported, replace all references to MAX_VIEWPORTS
    and GetFloati_v with MAX_VIEWPORTS_NV and GetFloati_vNV respectively.
    If OES_viewport_array is supported, replace all references to MAX_VIEWPORTS
    and GetFloati_v with MAX_VIEWPORTS_OES and GetFloati_vOES respectively.

Issues

    (1) Does this extension provide any functionality to convert the w-scaled
        image to the barrel distorted image used in VR?

      RESOLVED: No. VR applications would still require a post-processing step to
      generate a barrel distorted image to compensate for the lens distortion.
      The following vertex and fragment shader pair un-warps a w-scaled image.
      It can be incorporated into an existing post-processing shader to directly
      convert a w-scaled image to the barrel distorted image.

        // Vertex Shader
        // Draw a triangle that covers the whole screen
        const vec4 positions[3] = vec4[3](vec4(-1, -1, 0, 1),
                                          vec4( 3, -1, 0, 1),
                                          vec4(-1,  3, 0, 1));
        out vec2 uv;
        void main()
        {
          vec4 pos = positions[ gl_VertexID ];
          gl_Position = pos;
          uv = pos.xy;
        }

        // Fragment Shader
        uniform sampler2D tex;
        uniform float xcoeff;
        uniform float ycoeff;
        out vec4 Color;
        in vec2 uv;

        void main()
        {
          // Handle uv as if upper right quadrant
          vec2 uvabs = abs(uv);

          // unscale: transform w-scaled image into an unscaled image
          //   scale: transform unscaled image int a w-scaled image
          float unscale = 1.0 / (1 + xcoeff * uvabs.x + xcoeff * uvabs.y);
          //float scale = 1.0 / (1 - xcoeff * uvabs.x - xcoeff * uvabs.y);

          vec2 P = vec2(unscale * uvabs.x, unscale * uvabs.y);

          // Go back to the right quadrant
          P *= sign(uv);

          Color = texture(tex, P * 0.5 + 0.5);
        }

    (2) In the standard use case a application sets up 4 viewports, one for
        each quadrant. Does each primitive have to be broadcast to all the 4
        viewports?

      RESOLVED: No. Applications may see a better performance if the viewport
      mask for each primitive is limited to the viewports corresponding
      to the quadrants it falls in.


Revision History

    Revision 1
      - Internal revisions.
    Revision 2
      - Add _NV suffixes to _COEFF tokens
    Revision 3
      - Add ES interactions.
      - Add requirement for NV_viewport_array2
    Revision 4, 2017/11/25 (pbrown)
      - Add to the OpenGL ES Extension Registry
