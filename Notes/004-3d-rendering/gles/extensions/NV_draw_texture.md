# NV_draw_texture

Name

    NV_draw_texture

Name Strings

    GL_NV_draw_texture

Contributors

    Steven Holte, NVIDIA Corporation (sholte 'at' nvidia.com)

Contact

    Pat Brown, NVIDIA Corporation (pbrown 'at' nvidia.com)

Status

    Complete

Version

    Last Modified Date:         9/19/2012
    NVIDIA Revision:            2

Number

    OpenGL Extension #430
    OpenGL ES Extension #126

Dependencies

    This extension is written against the OpenGL 4.1 Specification
    (Compatibility Profile).

    This extension can also be used with OpenGL ES 2.0 or later (see the section,
    "Interactions with OpenGL ES," below).

    This extension interacts with EXT_shadow_samplers.

Overview

    This extension provides a new function, DrawTextureNV(), allowing
    applications to draw an screen-aligned rectangle displaying some or all of
    the contents of a two-dimensional or rectangle texture.  Callers specify a
    texture object, an optional sampler object, window coordinates of the
    rectangle to draw, and texture coordinates corresponding to the corners of
    the rectangle.  For each fragment produced by the rectangle, DrawTextureNV
    interpolates the texture coordinates, performs a texture lookup, and uses
    the texture result as the fragment color.

    No shaders are used by DrawTextureNV; the results of the texture lookup
    are used in lieu of a fragment shader output.  The fragments generated are
    processed by all per-fragment operations.  In particular,
    DrawTextureNV() fully supports blending and multisampling.

    While this functionality can be obtained in unextended OpenGL by drawing a
    rectangle and using a fragment shader to do a texture lookup,
    DrawTextureNV() is likely to have better power efficiency on
    implementations supporting this extension.  Additionally, use of this
    extension frees the application developer from having to set up
    specialized shaders, transformation matrices, vertex attributes, and
    various other state in order to render the rectangle.

New Procedures and Functions

    void DrawTextureNV(GLuint texture, GLuint sampler,
                       GLfloat x0, GLfloat y0, 
                       GLfloat x1, GLfloat y1,
                       GLfloat z,
                       GLfloat s0, GLfloat t0, 
                       GLfloat s1, GLfloat t1);

New Tokens

    None.

Additions to Chapter 2 of the OpenGL 4.1 Specification (OpenGL Operation)

    Modify Section 2.19, Conditional Rendering, p. 183

    (modify first paragraph to specify that DrawTextureNV is affected by
    conditional rendering) ... is false, all rendering commands between
    BeginConditionalRender and the corresponding EndConditionalRender are
    discarded.  In this case, Begin, End, ...and DrawTextureNV (section 4.3.X)
    have no effect.


Additions to Chapter 3 of the OpenGL 4.1 Specification (Rasterization)

    Modify Section 3.1, Discarding Primitives Before Rasterization, p. 204

    (modify the end of the second paragraph) When enabled, RASTERIZER_DISCARD
    also causes the [[compatibility profile only:  Accum, Bitmap, CopyPixels,
    DrawPixels,]] Clear, ClearBuffer*, and DrawTextureNV commands to be
    ignored.

Additions to Chapter 4 of the OpenGL 4.1 Specification (Per-Fragment
Operations and the Frame Buffer)

    (Insert new section after Section 4.3.1, Writing to the Stencil or
    Depth/Stencil Buffers, p. 380)

    Section 4.3.X, Drawing Textures

    The command:

      void DrawTextureNV(GLuint texture, GLuint sampler,
                         GLfloat x0, GLfloat y0, 
                         GLfloat x1, GLfloat y1,
                         GLfloat z,
                         GLfloat s0, GLfloat t0, 
                         GLfloat s1, GLfloat t1);

    is used to draw a screen-aligned rectangle displaying a portion of the
    contents of the texture <texture>.  The four corners of this
    screen-aligned rectangle have the floating-point window coordinates
    (<x0>,<y0>), (<x0>,<y1>), (<x1>,<y1>), and (<x1>,<y0>).  A fragment will
    be generated for each pixel covered by the rectangle.  Coverage along the
    edges of the rectangle will be determined according to polygon
    rasterization rules.  If the framebuffer does not have a multisample
    buffer, or if MULTISAMPLE is disabled, fragments will be generated
    according to the polygon rasterization algorithm described in section
    3.6.1.  Otherwise, fragments will be generated for the rectangle using the
    multisample polygon rasterization algorithm described in section 3.6.6.
    In either case, the set of fragments generated is not affected by other
    state affecting polygon rasterization -- in particular, the CULL_FACE,
    POLYGON_SMOOTH, and POLYGON_OFFSET_FILL enables and PolygonMode state have
    no effect.  All fragments generated for the rectangle will have a Z window
    coordinate of <z>.

    The color associated with each fragment produced will be obtained by using
    an interpolated source coordinate (s,t) to perform a lookup into <texture>
    The (s,t) source coordinate for each fragment is interpolated over the
    rectangle in the manner described in section 3.6.1, where the (s,t)
    coordinates associated with the four corners of the rectangle are:

      (<s0>, <t0>) for the corner at (<x0>, <y0>),
      (<s1>, <t0>) for the corner at (<x1>, <y0>),
      (<s1>, <t1>) for the corner at (<x1>, <y1>), and
      (<s0>, <t1>) for the corner at (<x0>, <y1>).

    The interpolated texture coordinate (s,t) is used to obtain a texture
    color (Rs,Gs,Bs,As) from the <texture> using the process described in
    section 3.9.  The sampler state used for the texture access will be taken
    from the texture object <texture> if <sampler> is zero, or from the
    sampler object given by <sampler> otherwise.  The filtered texel <tau> is
    converted to an (Rb,Gb,Bb,Ab) vector according to table 3.25 and swizzled
    as described in Section 3.9.16.  [[Core Profile Only:  The section
    referenced here is present only in the compatibility profile; this
    language should be changed to reference the relevant language in the core
    profile.]]

    The fragments produced by the rectangle are not processed by fragment
    shaders [[Compatibility Profile:  or fixed-function texture, color sum, or
    fog operations]].  These fragments are processed by all of the
    per-fragment operations in section 4.1.  For the purposes of the scissor
    test (section 4.1.2), the enable and scissor rectangle for the first
    element in the array of scissor test enables and rectangles are used.

    The error INVALID_VALUE is generated by DrawTextureNV if <texture> is not
    the name of a texture object, or if <sampler> is neither zero nor the name
    of a sampler object.  The error INVALID_OPERATION is generated if the
    target of <texture> is not TEXTURE_2D or TEXTURE_RECTANGLE, <texture> is
    not complete, if <sampler> is zero and the TEXTURE_COMPARE_MODE parameter
    of <texture> is COMPARE_REF_TO_TEXTURE, or if <sampler> is non-zero and
    the TEXTURE_COMPARE_MODE_PARAMETER of <sampler> is COMPARE_REF_TO_TEXTURE.


Additions to Chapter 5 of the OpenGL 4.1 Specification (Special Functions)

    None.

Additions to Chapter 6 of the OpenGL 4.1 Specification (State and
State Requests)

    None.

Additions to Appendix A of the OpenGL 4.1 Specification (Invariance)

    None.

Additions to the AGL/GLX/WGL Specifications

    None.

GLX Protocol

    !!! TBD

Errors

    INVALID_VALUE is generated by DrawTextureNV if <texture> is not the name
    of a texture object, or if <sampler> is neither zero nor the name of a
    sampler object.

    INVALID_OPERATION is generated by DrawTextureNV if the target of <texture>
    is not TEXTURE_2D or TEXTURE_RECTANGLE, <texture> is not complete, if
    <sampler> is zero and the TEXTURE_COMPARE_MODE parameter of <texture> is
    COMPARE_REF_TO_TEXTURE, or if <sampler> is non-zero and the
    TEXTURE_COMPARE_MODE_PARAMETER of <sampler> is COMPARE_REF_TO_TEXTURE.

New State

    None.


New Implementation Dependent State

    None.

Interactions with OpenGL ES

    If implemented for OpenGL ES, NV_draw_texture acts as described in this spec,
    except:

        * Ignore the references to conditional rendering including changes to
          section 2.19 "Conditional Rendering".
        * Ignore all references to RASTERIZER_DISCARD including changes to
          section 3.1 "Discarding Primitives Before Rasterization".
        * Ignore references to MULTISAMPLE.
        * Ignore references to POLYGON_SMOOTH and PolygonMode.
        * Ignore references to TEXTURE_RECTANGLE.
        * If the version of OpenGL ES is less than 3.0, the sampler parameter
          must always be 0.
        * If the version of OpenGL ES is less than 3.0, ignore references to
          texture swizzles.

Interactions with OpenGL ES and EXT_shadow_samplers

    If implemented for OpenGL ES with the EXT_shadow_samplers extension,
    replace references to TEXTURE_COMPARE_FUNC, TEXTURE_COMPARE_MODE, and
    COMPARE_REF_TO_TEXTURE, with references to TEXTURE_COMPARE_FUNC_EXT,
    TEXTURE_COMPARE_FUNC_EXT and COMPARE_REF_TO_TEXTURE_EXT.

    If implemented for OpenGL ES without the EXT_shadow_samplers extension,
    ignore references to these symbols.

Issues

    (1) Why provide this extension when you can do the same thing by drawing a
        quad with a simple fragment shader using texture mapping?

      RESOLVED:  This extension is intended to provide a high-performance
      power-efficient fixed-function path for drawing the contents of a
      texture onto the screen.  No vertex shader is required to position the
      vertices of the quad, and no fragment shader is required to perform a
      texture lookup.

    (2) Why provide this extension when you can do something similar with
        DrawPixels?

      RESOLVED:  DrawPixels provides similar functionality, but can only
      access client memory or a pixel buffer object.  If the data to be drawn
      on-screen come from a texture, it would be necessary to read the
      contents of the texture back to client memory or a pixel buffer object
      before drawing.  

      Additionally, the rendering process for DrawPixels has several
      limitations.  Addressing a subset of the source data requires either
      pointer manipulation or the use of the separate PixelStore APIs, and
      doesn't permit sub-pixel addressing in the source data.  While
      DrawPixels supports scaling via the PixelZoom, the zooming capability
      provides only point-sampled filtering.  Additionally, DrawPixels is not
      supported in the core profile of OpenGL, or in OpenGL ES.

    (3) Why provide this extension when you can do something similar with
    BlitFramebuffer?

      RESOLVED:  BlitFramebuffer also provides similar functionality, but it
      does not permit per-fragment operations like blending, which is a
      significant limitation for some important "2D" use cases of this API
      (e.g., compositing several images from textures).  Additionally, need to
      attach the texture to a framebuffer object, set up a read buffer, and
      bind the framebuffer object as the read framebuffer result in several
      additional steps not present in the DrawTextureNV API.

    (4) The DrawTextureNV API only supports 2D or rectangle textures.  Should
        we provide support for accessing other types of texture (1D, 3D, cube
        maps, arrays)?  Or even for pulling a "2D" image out of a more complex
        texture (like identifying a texture face, or a layer of a 2D array
        texture or a 3D texture)?

      RESOLVED:  No, we are choosing to keep the API simple and support only
      2D/rectangle textures.  Adding in support for 3D or array textures would
      require additional texture coordinates that would clutter up the "2D"
      API or a separate "DrawTexture3DNV" API taking (s,t,r) coordinates.
      Adding in support for pulling out a face/layer of a texture with
      multiple layers would inject similar clutter or new APIs.

      Note that the face/layer selection could also be handled by a
      Direct3D-like "resource view" API that would allow callers to create
      multiple "views" of a source texture.  In particular, one might be able
      to use such an extension to create a "virtual" 2D texture object that
      refers to a single face/layer of a cube map, 2D array, or 3D texture.

    (5) Should we support multisample textures (TEXTURE_2D_MULTISAMPLE)?

      RESOLVED:  No.  Current texture mapping support for multisample texture
      only allows for selection of a single numbered texture.  There are no
      filtered texture lookup capabilities for these sorts of textures.

      BlitFramebuffer does support sourcing a multisample texture (via a
      framebuffer object attachement), but its capabilities are also fairly
      limited -- copies are only supported either by first resolving multiple
      samples down to a single sample, or doing a straight sample-by-sample
      copy to a matching multisample buffer.

    (6) What sort of coordinates should be used to access the texture?

      RESOLVED:  We use the same coordinate system as is used for normal
      texture lookups for a given texture target.  

      For textures with a TEXTURE_RECTANGLE target, we use non-normalized
      coordinates -- to draw a 640x480 rectangle texture on top of a 640x480
      window, you would call:

        glDrawTexture(texture, sampler, 
                      0, 0, 640, 480,  /* destination */
                      0, 0, 640, 480   /* texture */);

      For textures with a TEXTURE_2D target, we use normalized coordinates.
      The same example as above with a 640x480 2D texture would use:

        glDrawTexture(texture, sampler, 
                      0, 0, 640, 480,  /* destination */
                      0, 0, 1, 1       /* texture */);

    (7) What limitations apply to the texture accesses in DrawTextureNV?

      RESOLVED:  We do not support any texture targets other than TEXTURE_2D
      and TEXTURE_RECTANGLE.  We also do not support shadow mapping via the
      TEXTURE_COMPARE_MODE parameter, given that we don't provide any
      interface for specifying a depth reference value.  In either case, an
      INVALID_OPERATION error will be generated if an unsupported feature is
      used.

    (8) Is anisotropic texture filtering supported?

      RESOLVED:  Yes.  However, anisotropic filtering may result in lower
      performance and power efficiency and should be used only if
      required. Given that the destination is a screen-aligned rectangle and
      the portion of texture sampled from is a texture-aligned rectangle, the
      footprints of pixels in texture space are regular.  Unless the
      DrawTextureNV command uses a non-uniform scale, anisotropic filtering
      should provide no benefit.

    (9) Are texture swizzles supported?

      RESOLVED:  Yes.

    (10) Does DrawTextureNV support multisample rasterization?

      RESOLVED:  Yes.  The coordinates of the destination rectangle are
      floating-point values, allowing for rectangle boundaries not on pixel
      edges.  When multisample rasterization is enabled, pixels on the edge of
      the rectangle may be partially covered, in which case only some samples
      of the pixel will be updated.  This multisample support allows for
      smoother panning of the drawn rectangles than one could get with the
      pixel-aligned updates provided by the BlitFramebuffer API.

    (11) Does DrawTextureNV support per-sample shading (i.e., a different
         color for each sample in the destination rectangle)?

      RESOLVED:  No.

    (12) Should any per-fragment operations be supported by this extension?

      RESOLVED:  Yes, we will all support fragment operations.  In particular,
      blending is particularly important for "2D" operations such as
      compositing image layers.  It seems interesting to allow stencil
      operations to "cut out" portions of the primitive.  It also seems
      interesting to allow depth testing be used to compare the DrawTextureNV
      rectangle (at a fixed depth) against previously rendered primitives
      (either "3D" or "2D").

    (13) Should we provide a mode to override/disable selected per-fragment
         operations when performing DrawTextureNV?

      RESOLVED:  No.  An override would be useful if we expected applications
      to be performing operations like toggling between regularly rendered
      primitives (with depth testing enabled) and "flat" DrawTexture2D output
      (not wanting depth testing) at a fine granularity.  It's not clear that
      such usage would be common.  If we expect switching between modes only
      at a coarse granularity, it would be simpler to require the application
      to apply the (infrequent) overrides themselves instead of adding clutter
      to the DrawTextureNV API.

    (14) Is it legal to call DrawTextureNV while transform feedback is active?
         If so, what is recorded?

      UNRESOLVED:  Yes, it's legal to call DrawTextureNV during transform
      feedback.  Nothing should be recorded in this case.  This is consistent
      with the handling of other "special" rendering operations (like
      DrawPixels and BlitFramebuffer).  This behavior falls out of the
      definition of transform feedback with no spec changes required; there
      are no geometric primitives sent through the pipeline for DrawTextureNV
      that could be recorded.

    (15) How does DrawTextureNV interact with RASTERIZER_DISCARD?

      UNRESOLVED:  If RASTERIZER_DISCARD is enabled, DrawTextureNV will be
      discarded.  This is consistent with the behavior of DrawPixels.  

      Note:  It appears that BlitFramebuffer is not affected by
      RASTERIZER_DISCARD, though the extensions that introduced this command
      don't explicitly address this one way or the other.

    (16) Should samples generated by DrawTextureNV be counted in occlusion
         queries?

      UNRESOLVED:  Yes.  Occlusion query is just another per-fragment
      operation, and we support all the other ones.

    (17) How does this extension interact with the DEPTH_CLAMP enable?

      UNRESOLVED:  When enabled, depth clamping will be performed on
      DrawTextureNV fragments.  This appears to be consistent with the spec
      language, as applied to DrawPixels.  There are two parts to depth
      clamping:  (a) clipping to the near/far frustum clip planes are
      disabled, and (b) clamping is applied to fragment Z as part of the depth
      test.  There's no language suggesting that (b) doesn't apply to color
      DrawPixels or Bitmap commands.  (DrawPixels with DEPTH_COMPONENT pixels
      is a different beast that doesn't go through the regular pixel path, and
      ARB_depth_clamp says that clamping doesn't apply there.)

      Note that if depth testing is disabled, the depth clamp enable has no
      effect on DrawTextureNV, since (a) doesn't apply because DrawTextureNV
      doesn't generate a geometric primitive that could be clipped.
      
    (18) How does the rectangle rendered by DrawTextureNV interact with
         polygon rasterization features (culling, polygon smooth, polygon
         mode, polygon offset)?

      RESOLVED:  None of these features affect DrawTextureNV.  The spec refers
      to the polygon rasterization of the spec only because we apply the same
      coverage computation rules to DrawTextureNV as are used for
      rasterization of single-sample and multisample polygons.

    (19) How does this extension interact with conditional rendering?

      UNRESOLVED:  If DrawTextureNV is called inside a BeginConditionalRender
      and EndConditionalRender pair and the query object indicates that
      rendering should be discarded, the DrawTextureNV command is also
      discarded.  This is consistent with the behavior of DrawPixels.


Revision History
    
    Rev.    Date    Author    Changes
    ----  --------  --------  -----------------------------------------
     1              pbrown    Internal revisions.
     2    09/19/12  sholte    Added ES interactions
