# OVR_multiview

Name

    OVR_multiview

Name Strings

    GL_OVR_multiview

Contact

    Cass Everitt, Oculus (cass.everitt 'at' oculus.com)

Contributors

    John Carmack, Oculus
    Tom Forsyth, Oculus
    Maurice Ribble, Qualcomm
    James Dolan, NVIDIA Corporation
    Mark Kilgard, NVIDIA Corporation
    Michael Songy, NVIDIA Corporation
    Yury Uralsky, NVIDIA Corporation
    Jesse Hall, Google
    Timothy Lottes, Epic
    Jan-Harald Fredriksen, ARM
    Jonas Gustavsson, Sony Mobile
    Sam Holmes, Qualcomm
    Nigel Williams, Qualcomm
    Tobias Hector, Imagination Technologies
    Daniel Koch, NVIDIA Corporation
    James Helferty, NVIDIA Corporation
    Olli Etuaho, NVIDIA Corporation

Status

    Complete.

Version

    Last Modified Date: October 19, 2018
    Revision: 6

Number

    OpenGL Extension #478
    OpenGL ES Extension #241

Dependencies

    OpenGL 3.0 or OpenGL ES 3.0 is required.

    This extension is written against the OpenGL ES 3.2 (November 3, 2016)
    specification and the OpenGL 4.6 (Core Profile) (July 30, 2017)

    This extension is written against the OpenGL ES Shading Language 3.00.04
    specification.

    This extension interacts with OpenGL 3.3, ARB_timer_query, and
    EXT_disjoint_timer_query.

    This extension interacts with OpenGL 4.5, ARB_direct_state_access and
    EXT_direct_state_access.

    This extension interacts with OpenGL ES 3.2, OpenGL 4.0,
    EXT_tessellation_shader, OES_tessellation_shader, and
    ARB_tessellation_shader.

    This extension interacts with OpenGL ES 3.2, OpenGL 3.2,
    EXT_geometry_shader, OES_geometry_shader and ARB_geometry_shader4

    This extension interacts with OpenGL 4.3, ARB_multi_draw_indirect, and
    EXT_multi_draw_indirect.

    This extension interacts with OpenGL 3.0, the OpenGL 3.0 Compatibility
    Profile, and ARB_fragment_layer_viewport.

Overview

    The method of stereo rendering supported in OpenGL is currently achieved by
    rendering to the two eye buffers sequentially.  This typically incurs double
    the application and driver overhead, despite the fact that the command
    streams and render states are almost identical.

    This extension seeks to address the inefficiency of sequential multiview
    rendering by adding a means to render to multiple elements of a 2D texture
    array simultaneously.  In multiview rendering, draw calls are instanced into
    each corresponding element of the texture array.  The vertex program uses a
    new gl_ViewID_OVR variable to compute per-view values, typically the vertex
    position and view-dependent variables like reflection.

    The formulation of this extension is high level in order to allow
    implementation freedom.  On existing hardware, applications and drivers can
    realize the benefits of a single scene traversal, even if all GPU work is
    fully duplicated per-view.  But future support could enable simultaneous
    rendering via multi-GPU, tile-based architectures could sort geometry into
    tiles for multiple views in a single pass, and the implementation could even
    choose to interleave at the fragment level for better texture cache
    utilization and more coherent fragment shader branching.

    The most obvious use case in this model is to support two simultaneous
    views: one view for each eye.  However, we also anticipate a usage where two
    views are rendered per eye, where one has a wide field of view and the other
    has a narrow one.  The nature of wide field of view planar projection is
    that the sample density can become unacceptably low in the view direction.
    By rendering two inset eye views per eye, we can get the required sample
    density in the center of projection without wasting samples, memory, and
    time by oversampling in the periphery.


New Tokens

    Accepted by the <pname> parameter of GetFramebufferAttachmentParameteriv:

        FRAMEBUFFER_ATTACHMENT_TEXTURE_NUM_VIEWS_OVR               0x9630
        FRAMEBUFFER_ATTACHMENT_TEXTURE_BASE_VIEW_INDEX_OVR         0x9632

    Accepted by the <pname> parameter of GetIntegerv:

        MAX_VIEWS_OVR                                              0x9631

    Returned by CheckFramebufferStatus:

        FRAMEBUFFER_INCOMPLETE_VIEW_TARGETS_OVR                    0x9633


New Procedures and Functions

    void FramebufferTextureMultiviewOVR( enum target, enum attachment,
                                         uint texture, int level,
                                         int baseViewIndex, sizei numViews );

    [[ If OpenGL 4.5 or ARB_direct_state_access is supported ]]

    void NamedFramebufferTextureMultiviewOVR( uint framebuffer, enum attachment,
                                              uint texture, int level,
                                              int baseViewIndex, sizei numViews );


Modifications to Chapter 4 of the OpenGL 4.6 Specification (Event Model)

    Modify section 4.3 (Time Queries) adding the following to the list
    of errors:

    "Queries where BeginQuery or EndQuery is called with a target of
    TIME_ELAPSED, or a if QueryCounter is called with a target of TIMESTAMP
    return undefined values if the draw framebuffer is multiview at any
    point during their execution."

Modifications to Chapter 9 of the OpenGL ES 3.2 Specification (Framebuffers
and Framebuffer Objects)

    Add a new subsection to section 9.2.2 (Attaching Images to Framebuffer
    Objects):

    "9.2.2.2 (Multiview Images)

    Finally, multiple layers of two-dimensional array textures can be
    attached to an attachment point. Such attachments represent multiple
    views, and the corresponding attachment point is considered to be
    _multiview_.

    In this mode there are several restrictions:

        - in vertex shader gl_Position is the only output that can depend on
          gl_ViewID_OVR (see Section 7.1 of the OpenGL ES Shading Language
          specification)
        - no transform feedback (section 11.1.3.11))
        - no tessellation control or evaluation shaders (section 11.1.3.11)
        - no geometry shader (section 11.1.3.11)
        - no timer queries (section 4.3)
        - occlusion query results must be between max per-view and the sum
          of the per-view queries, inclusive (section 15.1.4)."
        - in fragment shader the contents of gl_Layer are undefined

    [[ If implemented in OpenGL ]]
        - the number of views rendered to by Begin/End is an undefined subset
          of the views present in the framebuffer

    Add the following to list of <pname> parameters which can be queried
    via GetFramebufferAttachmentParameteriv when the value of
    FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE is TEXTURE in section 9.2.3
    (Framebuffer Object Queries):

    "If <pname> is FRAMEBUFFER_ATTACHMENT_TEXTURE_NUM_VIEWS_OVR and the value
    of FRAMEBUFFER_ATTACHMENT_OBJECT_NAME is a two-dimensional array texture,
    then <params> will contain the number of views that were specified for the
    attachment point via FramebufferTextureMultiviewOVR. Otherwise, <params>
    will contain zero.

    "If <pname> is FRAMEBUFFER_ATTACHMENT_TEXTURE_BASE_VIEW_INDEX_OVR and the
    value of FRAMEBUFFER_ATTACHMENT_OBJECT_NAME is a two-dimensional array
    texture, then <params> will contain the base view index that was specified
    for the attachment point via FramebufferTextureMultiviewOVR. Otherwise,
    <params> will contain zero."

    Add the following to the end of section 9.2.8 (Attaching Textures to a
    Framebuffer), immediately before the subsection "Effects of Attaching a
    Texture Image":

    "Multiple layers of a two-dimensional array texture can be
    attached as one of the logical buffers of a framebuffer object with the
    commands

        void FramebufferTextureMultiviewOVR( enum target, enum attachment,
                                             uint texture, int level,
                                             int baseViewIndex, sizei numViews );

      [[ If OpenGL 4.5 or ARB_direct_state_access is supported ]]

        void NamedFramebufferTextureMultiviewOVR( uint framebuffer, enum attachment,
                                                  uint texture, int level,
                                                  int baseViewIndex, sizei numViews );

    These commands operate similarly to the FramebufferTextureLayer and
    NamedFramebufferTexture commands, except that <baseViewIndex>
    and <numViews> select a range of texture array elements that will be
    targeted when rendering. Such an attachment is considered _multiview_
    (section 9.2.2.2) and rendering commands issued when such a framebuffer
    object is bound are termed "multiview rendering". The maximum number
    of views which can be bound simultaneously is determined by the value
    of MAX_VIEWS_OVR, which can be queried with the GetIntegerv command.

    The command

        View( uint id );

    does not exist in the GL, but is used here to describe the multiview
    functionality in this section.  The effect of this hypothetical function
    is to set the value of the shader built-in input gl_ViewID_OVR.

    When multiview rendering is enabled, the Clear (section 15.2.3),
    ClearBuffer* (section 15.2.3.1), and Draw* (section 10.5)
    commands have the same effect as:

        for( int i = 0; i < numViews; i++ ) {
            for ( enum attachment : all attachment values where multiple texture array elements have been targeted for rendering ) {
                FramebufferTextureLayer( target, attachment, texture, level, baseViewIndex + i );
            }
            View( i );
            <command>
        }

    The result is that every such command is broadcast into every active
    view. The shader uses gl_ViewID_OVR to compute view dependent outputs.

    The number of views, as specified by <numViews>, must be the same for all
    framebuffer attachments points where the value of FRAMEBUFFER_ATTACHMENT_-
    OBJECT_TYPE is not NONE or the framebuffer is incomplete (section 9.4.2).

    If <texture> is non-zero and the command does not result in an error, the
    framebuffer attachment state corresponding to <attachment> is updated as
    in the FramebufferTextureLayer command, except that the values of
    FRAMEBUFFER_ATTACHMENT_TEXTURE_LAYER and
    FRAMEBUFFER_ATTACHMENT_TEXTURE_BASE_VIEW_INDEX_OVR are is set to
    <baseViewIndex>, and the value of
    FRAMEBUFFER_ATTACHMENT_TEXTURE_NUM_VIEWS_OVR is set to <numViews>.

    Errors

    In addition to the corresponding errors for FramebufferTextureLayer when
    called with the same parameters (other than <layer>):

    An INVALID_VALUE error is generated if:
    - <numViews> is less than 1 or if <numViews> is greater than MAX_VIEWS_OVR.
    - <texture> is a two-dimensional array texture and <baseViewIndex> +
      <numViews> is larger than the value of MAX_ARRAY_TEXTURE_LAYERS.
    - texture is non-zero and <baseViewIndex> is negative."

    An INVALID_OPERATION error is generated if texture is non-zero and is not
    the name of a two-dimensional array texture."

    Add the following to the list of conditions required for framebuffer
    attachment completeness in section 9.4.1 (Framebuffer Attachment
    Completeness):

    "If <image> is a two-dimensional array and the attachment
    is multiview, all the selected layers, [<baseViewIndex>,
    <baseViewIndex> + <numViews>), are less than the layer count of the
    texture."

    Add the following to the list of conditions required for framebuffer
    completeness in section 9.4.2 (Whole Framebuffer Completeness):

    "The number of views is the same for all populated attachments.

    { FRAMEBUFFER_INCOMPLETE_VIEW_TARGETS_OVR }"

Modifications to Chapter 11 of the OpenGL ES 3.2 Specification (Programmable
Vertex Processing

    Modify section 11.1.3.11 (Validation) adding the following conditions
    to the list of reasons that may result in an INVALID_OPERATION error
    being generated by any command that transfers vertices to the the GL:

    * Any attachment of the draw framebuffer is multiview (section 9.2.8)
      and any of the following conditions are true:

      - There is an active program for tessellation control, tessellation
      evaluation, or geometry stages, or

      - Transform feedback is active and not paused."

Modifications to Chapter 15 of the OpenGL ES 3.2 Specification (Writing
Fragments and Samples to the Framebuffer) [OpenGL 4.6 Chapter 17]

    Modify section 15.1.4 [OpenGL 4.6 section 17.3.5] (Occlusion Queries)
    adding the following to the end of the second paragraph (describing
    the SAMPLES_PASSED query):

    "During multiview rendering (section 9.2.8), the samples-passed count
    requirement is relaxed and the samples counted must be between the
    maximum number of samples counted from any view, and the sum of samples
    counted for all views."

    Add the following to the end of the fourth paragraph (describing
    ANY_SAMPLES_PASSED and ANY_SAMPLES_PASSED_CONSERVATIVE):

    "During multiview rendering (section 9.2.8), the samples-boolean state
    is set to TRUE if the samples-boolean state from any view is set to
    TRUE."

Modifications to Chapter 16 of the OpenGL ES 3.2 Specification (Reading and
Copying Pixels)

    Add the following paragraph to the end of the description of
    BlitFramebuffer in section 16.2.1 (Blitting Pixel Rectangles):

    "If the draw framebuffer has multiple views (see section 9.2.8,
    FramebufferTextureMultiviewOVR), values taken from the read buffer are
    only written to draw buffers in the first view of the draw framebuffer."


New Implementation Dependent State

    (Additions to Table 21.40 "Implementation Dependent Values")

    Get Value      Type  Get Command  Minimum Value  Description              Sec.
    ---------      ----  -----------  -------------  -----------              ----
    MAX_VIEWS_OVR   Z+   GetIntegerv  2              Maximum number of views  9.2.8

Modifications to The OpenGL ES Shading Language Specification, Version 3.00.04

    Including the following line in a shader can be used to control the
    language features described in this extension:

      #extension GL_OVR_multiview : <behavior>

    where <behavior> is as specified in section 3.5.

    A new preprocessor #define is added to the OpenGL ES Shading Language:

      #define GL_OVR_multiview 1


    In section 4.3.8.1 "Input Layout Qualifiers":

    Insert a paragraph before the final one ("Fragment shaders cannot ..."):

    Vertex shaders also allow the following layout qualifier on "in" only
    (not with variable declarations)

    [[ If implemented in OpenGL ES]]

          layout-qualifier-id
                num_views = integer-constant

    [[ If implemented in OpenGL ]]

          layout-qualifier-id
                num_views = integer-constant-expression

    to indicate that the shader will only be used with the given number of
    views, as described in section 4.4 ("Framebuffer Objects") of the OpenGL ES
    Specification.  If this qualifier is not declared, the behavior is as if it
    had been set to 1.

    If this layout qualifier is declared more than once in the same shader,
    all those declarations must set num_views to the same value; otherwise a
    compile-time error results. If multiple vertex shaders attached to a
    single program object declare num_views, the declarations must be
    identical; otherwise a link-time error results. It is a compile-time
    error to declare num_views to be less than or equal to zero, or greater
    than MAX_VIEWS_OVR.

    Additions to Section 7.1 "Built-in Language Variables"

    Add the following to the list of built-in variables that are intrinsically
    declared in the vertex and fragment shading languages:

       in mediump uint gl_ViewID_OVR;

    The gl_ViewID_OVR built-in variable holds the integer index of the view
    number to which the current shader invocation belongs, as defined in
    in section 4.4.2.4 (FramebufferTextureMultiviewOVR) in the OpenGL
    Graphics Systems Specification.

    [[ If OVR_multiview2 is not supported ]]

    It is a compile- or link-time error if any output variable other
    than gl_Position is statically dependent on gl_ViewID_OVR. If an
    output variable other than gl_Position is dynamically dependent on
    gl_ViewID_OVR, the values are undefined.

    NOTE: Implementations that also support OVR_multiview2 may not
    generate an error if these conditions are violated, even if the
    OVR_multiview2 extension is not enabled.

Errors

    INVALID_FRAMEBUFFER_OPERATION is generated by commands that read from the
    framebuffer such as BlitFramebuffer, ReadPixels, CopyTexImage*, and
    CopyTexSubImage*, if the number of views in the current read framebuffer
    is greater than 1.

    INVALID_OPERATION is generated if a rendering command is issued and the the
    number of views in the current draw framebuffer is not equal to the number
    of views declared in the currently bound program.

Interactions with OpenGL 3.3, ARB_timer_query, and EXT_disjoint_timer_query

    If none of OpenGL 3.3, ARB_timer_query, or EXT_disjoint_timer_query
    are supported, ignore references to TIMESTAMP and TIME_ELAPSED queries.

Interactions with OpenGL 4.5, ARB_direct_state_access, and
EXT_direct_state_access

    If none of OpenGL 4.5, ARB_direct_state_access and EXT_direct_state_access
    are supported, the command NamedFramebufferTextureMultiviewOVR does not
    exist.

Interactions with OpenGL ES 3.2, OpenGL 4.0, EXT_tessellation_shader,
OES_tessellation_shader and ARB_tessellation_shader.

    If none of OpenGL ES 3.2, OpenGL 4.0, EXT_tessellation_shader,
    OES_tessellation_shader or ARB_tessellation shader are supported, ignore
    all references to tessellation shaders.

Interactions with OpenGL ES 3.2, OpenGL 3.2, EXT_geometry_shader,
OES_geometry_shader, and ARB_geometry_shader4.

    If none of OpenGL ES 3.2, OpenGL 3.2, EXT_geometry_shader,
    OES_geometry_shader, or ARB_geometry_shader4 are supported, ignore all
    references to geometry shaders.

Interactions with OpenGL 4.3, ARB_multi_draw_indirect, and
EXT_multi_draw_indirect.

    If none of OpenGL 4.3, ARB_multi_draw_indirect, or EXT_multi_draw_indirect
    are supported, ignore all references to multi-draw-indirect.

Interactions with OpenGL 3.0

    If OpenGL 3.0 (or later) is not supported, ignore all references to the
    SAMPLES_PASSED occlusion query target.

Interactions with OpenGL ES 3.2, OpenGL 4.3, EXT_geometry_shader,
OES_geometry_shader, and ARB_fragment_layer_viewport

    If none of OpenGL ES 3.2, OpenGL 4.3, EXT_geometry_shader,
    OES_geometry_shader, or ARB_fragment_layer_viewport is supported, ignore
    all references to gl_Layer.

Interactions with OpenGL 3.0 Compatibility Profile

    If OpenGL 3.0 Compatibility Profile (or later) is not supported, ignore all
    references to Begin/End.


Issues

    (1) Should geometry shaders be allowed in multiview mode?

    Resolved: Not in this extension. By disallowing it, we hope to enable more
    implementations to be available sooner, and there are complex issues that
    arise when a GS is used to target gl_Layer explicitly.

    (2) Should there be separate scissor and viewport per view?

    Resolved: No, while there might be some uses for such support, it adds
    unnecessary implementation complexity. In the case of inset rendering, there
    will be a need to adjust the scissor per view.  We will defer that issue for
    now by forcing all views to use the same scissor and viewport.

    (3) Why not just use geometry shaders?

    Resolved: GS could be used to achieve stereo rendering by broadcasting each
    primitive to each view.  The problem with this approach is that it requires
    the GS's very general mechanism with known performance implications to solve
    a problem that does not require that solution and perhaps more frequently
    than not, will not be the most efficient means of implementation.

    (4) Why use texture arrays instead of separate FBOs?

    Resolved: Use of arrays does imply that we use a minimum version of GL and
    ES 3.0. On the other hand, it has some nice simplifying properties.  In
    particular, the format and resolution of each view is known to be the same
    and only one FBO is bound, just like with normal rendering.  It has some
    potentially limiting interactions with GS use, but on the whole, the
    implementation simplifications are considered worth the implied limitations.

    (5) How does this extension interact with occlusion queries, timer queries,
    etc?

    Resolved: The bias will be toward relaxed rules to allow implementation
    freedom. For example, occlusion queries should not return fewer than the max
    samples returned from any view, but returning the sum may also be fine.
    Simply reporting the result from view 0 is not sufficient.

    (6) Is gl_ViewID_OVR visible at every pipeline stage?

    Resolved: To make integration simple for app developers, the intent is for
    gl_ViewID_OVR to be visible as a built-in at each programmable pipeline stage.

    (7) Are view-dependent parameters exposed explicitly?

    Resolved: No.  This is implicit in the dependence of a parameter on ViewID.
    In this extension, however, only gl_Position is allowed to depend on ViewID
    in the vertex shader. If a shader violates this restriction it results in
    undefined behavior or may produce an error.  Later extensions may relax that
    restriction.

    (8) Should the parameters that affect view-dependent position be driver
    visible or otherwise restricted?

    Resolved: No. Letting the app index via ViewID makes it a lot harder for the
    driver to know much about the details of the transform inputs that result in
    view-dependent outputs, but no such support exists in the API today either.
    Adding that support would be complex, and thus not in the spirit of an
    extension that can be implemented broadly with low risk.

    (9) Should there be a per-view UBO instead of exposing gl_ViewID_OVR in the shader?

    Resolved: No. Exposing the gl_ViewID_OVR variable is the smallest change. It
    does imply dynamic branching, predication, or indexing based on the view id,
    however, implementations could compile separate versions of the shader with
    view id translated to literals if that would improve performance.  Extra API
    and machinery would be required to sequence one or more UBOs to different
    objects per view.

    (10) Should there be DSA style entry points?

    Resolved: Almost certainly in GL 4.5 and beyond.  Less clear that it should
    be required for ES 3.0.

    (11) Should tessellation shaders be allowed in multiview mode?

    Resolved: Not in this extension. By disallowing it, we hope to enable more
    implementations to be available sooner.

    (12) Does the number of views need to be declared in the shader?

    Resolved: Yes. This enables the implementation to specialize the compilation
    for the declared number of views, for example by loop-unrolling. It could
    also serves as a notification that the shader outputs must be 'broadcast' to
    a set of views - even in cases where the shader does not reference from
    gl_ViewID_OVR.

    (13) What should happen if the number of views declared in the shader does
    not match num_views?

    Resolved: Such a mismatch can only be detected at draw call time, so it
    would have to either generate a draw call time error, or just result in
    undefined results for any view > num_views. To avoid undefined results, the
    draw call time error is preferred.

    (14) How should read operations on FBOs with multiview texture attachments be
    handled?

    Resolved: Generating INVALID_OPERATION when the target is GL_READ_FRAMEBUFFER
    for FramebufferTextureMultiviewOVR to disallow attaching a multiview texture
    to the current read framebuffer is not sufficient to prevent a read operation
    on a FBO with a multiview texture attachment. The multiview texture may still
    be attached to the current write framebuffer and then bound as the read
    framebuffer. Instead, an INVALID_FRAMEBUFFER_OPERATION error should be
    generated for any read operations that occur whenever a multiview texture is
    attached to the current read framebuffer. The multiview texture needs to be
    instead attached as a 2D texture array with the level explicitly specified to
    be read. This conforms to the expected behavior for read operations on FBOs
    with multiview texture attachments to be consistent with
    FramebufferTextureMultisampleMultiviewOVR.
    
    (15) How do clears apply to framebuffers with multiple views
    (bugzilla 16173)?
    
    Resolved: Clears are applied to all views.
    
    (16) How should blit operations to draw framebuffers with multiple views
    be handled (bugzilla 16174)?
    
    Resolved: The options are to broadcast the blit, only blit to a subset
    of views, or throw an error. There's no particularly compelling use case
    for either of the first options, so an error would have been desirable.
    However, this was missed when the extension was initially drafted, and
    implementations all ended up doing the same thing - blitting to just the
    first view of the draw framebuffer.
    
    (17) How do clears apply to framebuffers with multiple views
    (bugzilla 16173)?
    
    Resolved: Clears are applied to all views.

    (18) What happens if the num_views layout qualifier has an invalid value
    like zero or something above MAX_VIEWS_OVR?  What happens if num_views
    is specified multiple times in the same shader, or in multiple vertex
    shaders (this is only possible in OpenGL)?

    RESOLVED: This is treated similarly to how layout local_size_{x,y,z} is
    handled in compute shaders:
    - num_views <= 0 or > MAX_VIEWS_OVR is a compile error
    - if declared multiple times in the same shader the values
      must be the same or a compile-time error results.
    - if declared in multiple shaders attached to the same program object
      a link-time error results.

    (19) Is the shader built-in variable really called gl_ViewID_OVR? The
    shading language convention does not normally include an underscore before
    the vendor suffix.

    RESOLVED: Yes it really is. It doesn't quite follow the standard extension
    conventions, but there were a number of implementations that already
    supported it before this was realized. We'll just leave it this way for
    compatibility.

    (20) Is there a gl_MaxViewsOVR shading language built-in constant?

    RESOLVED: No. Initial implementations didn't provide this. If this ever
    is made KHR or core functionality, this could be added.

    (21) What is the minimum values permitted for MAX_VIEWS_OVR?

    RESOLVED: 2. But at least 6 is recommended for consistency with Vulkan.
    Six views is desireable as it can be used to render all faces of a cube map
    in a single pass. Added state table entries for this in revision 0.8.

    (22) What are the rules to check if anything other than gl_Position
    depends on gl_ViewID_OVR and what happens if the rules aren't met?

    RESOLVED: It is a compile- or link-time error if it can be determined
    that an output variable other than gl_Position is statically dependent on
    gl_ViewID_OVR. However, if an output variable other than gl_Position
    is dynamically dependent on gl_ViewID_OVR, the results are undefined.

    NOTE: This restriction is relaxed if OVR_multiview2 is supported,
    and some implementations may not implement these checks even if
    OVR_multiview2 is not enabled. It is recommended and preferred that
    all implemenations support OVR_multiview2 and that applications
    enable the extension when present.

    For non-VTG stages (eg fragment) which don't have a gl_Position output
    variable this means that no outputs can depend on the gl_VIewID_OVR.
    It is still possible that shader side effects (such as image or buffer
    stores, if supported) could be view-dependent.

    (23) What is the behaviour if transform feedback, tessellation or
    geometry shaders, or timer queries are used?

    RESOLVED: INVALID_OPERATION is generated for any draw operation
    with when the draw framebuffer is multiview if there is an active
    tessellation control, tessellation evaluation, or geometery shader,
    or if transform feedback is active.

    For timer queries there is no good time to do a multiview error check,
    (because a multiview framebuffer could be bound before or after the
    timer query has started) and thus the results of the TIME_ELAPSED
    and TIMESTAMP queries have undefined values if a multiview framebuffer
    was bound at any time during their execution.

    (24) Do instanced drawing commands (DrawArraysInstanced*, and
    DrawElementsInstanced*) work with this extension?

    RESOLVED: Yes. No specific edits are required because "It Just Works"(TM)

    (25) Do indirect drawing commands (DrawArraysIndirect, DrawElementsIndirect),
    work with this extension?

    RESOLVED: Yes. No specific edits are required because "It Just Works"(TM)

    (26) Do multi draw indirect commands (MultiDrawArraysIndirect,
    MultiDrawElementsIndirect) work with this extension?

    RESOLVED: Yes. No specific edits are required because "It Just Works"(TM)

    (27) What happens if the <baseViewIndex> + <numViews> is out of range?
    For example, if it's greater than the number of layers in the attached
    texture or greater than MAX_ARRAY_TEXTURE_LAYERS?

    RESOLVED: In cases there the resulting value could never be a valid
    <layer> argument to FramebufferTextureLayer, the INVALID_VALUE value
    is generated. This follows from the fact that
    FramebufferTextureMultiviewOVR is defined in terms of
    FramebufferTextureLayer. For cases which are dependant on the properties
    of a specific texture (referencing more layers than exist), this is a
    to be a framebuffer attachment completeness check as this can change if
    the texture is redefined after the FramebufferTextureMultiviewOVR call.

    (28) What is the value of FRAMEBUFFER_ATTACHMENT_TEXTURE_LAYER after
    FramebufferTextureMultiviewOVR is called?

    RESOLVED: It is set it to the value of <baseViewIndex>.
    A FRAMEBUFFER_ATTACHMENT_MULTIVIEW_OVR framebuffer property would be
    useful here..

    (29) Should there be a FRAMEBUFFER_ATTACHMENT_MULTIVIEW_OVR property to
    indicate multiview framebuffers, similar to the
    FRAMEBUFFER_ATTACHMENT_LAYERED property?

    RESOLVED: No. Ideally there would to allow distinguising between
    layered framebuffers and multiview framebuffers, and to know which of
    FRAMEBUFFER_ATTACHMENT_TEXTURE_LAYER,
    FRAMEBUFFER_ATTACHMENT_TEXTURE_NUM_VIEWS_OVR, and
    FRAMEBUFFER_ATTACHMENT_TEXTURE_BASE_VIEW_INDEX_OVR are valid.
    Unfortunately existing implementations don't have this, so it will
    just have to be implied. This is something that could be improved in
    a promoted version of this extension.

    (30) What types of textures can be used for multiview rendering?
    Two-dimensional array textures at a minimum, what about three-
    dimensional, cube map, cube map array or two-dimensional
    multisample textures which are also accepted by FramebufferTextureLayer
    (which is what this extension is defined in terms of).

    RESOLVED. The initial extension is limited to two-dimensional array
    textures. A future extension could extend it further if there is demand.

    (31) Is multiview supported for compute shaders? If so, how would it work?

    RESOLVED. No. Multiview rendering is orthogonal to compute shaders since
    Dispatch* commands do not operate on framebuffer attachments and therefore
    it is meaningless to try to describe interactions with multiview
    framebuffers. Prior to revision 3, this specification stated that
    gl_ViewID_OVR was available in the compute shading language but not all
    implementations did so, and for those that did, it is impossible for it
    to ever have a value other than 0.

    (32) If geometry and tessellation shaders are not supported with multiview
    rendering why allow gl_ViewID_OVR to be accepted in tessellation and
    geometry shaders?

    RESOLVED. There is no benefit. This was removed in revision 3. Older
    drivers may of course continue to allow such shaders to compile, but
    it is not possible to use them.

    (33) How does the gl_Layer builtin input in the fragment shader interact
    with multiview?

    RESOLVED: Whenever a multiview framebuffer is bound, the contents of
    gl_Layer in the fragment shader are undefined. This is consistent with
    issue (1), which disallows geometry shader with multiview framebuffers.

    (34) How does this extension interact with the OpenGL Compatibility
    Profile?

    RESOLVED: When a multiview framebuffer is bound, Begin/End may render to a
    subset of the views in the framebuffer. (Specifically which views is
    undefined, and may be any subset of the views attached to the framebuffer,
    including the null set.)


Revision History

      Rev.    Date    Author    Changes
      ----  --------  --------  -----------------------------------------
      0.1   10/17/14  cass      Initial draft
      0.2   11/10/14  cass      Changes to use texture array instead of separate FBOs
      0.3   02/05/15  cass      Only gl_Position can be view dependent in the vertex shader.
      0.4   02/11/15  cass      Switch to view instead of layer, as these are distinct now
      0.5   04/15/15  cass      Clean up pass before publishing
      0.6   07/01/16  nigelw    Modify errors to conform to multisample multiview spec changes
      0.7   07/25/17  tjh       Clarify Blit and Clear behaviours, and fixed framebuffer attachment
                                binding psuedocode (bugzillas 16173, 16174, 16176)
      1     10/10/17  dgkoch    Completed extension.
                                - Rebased on ES 3.2 (and OpenGL 4.6 where necessary) and added
                                  extension interactions
                                - Added DSA command for GL: NamedFramebufferTextureMultiviewOVR
                                - properly documented gl_ViewID_OVR and behaviour
                                - documented error behavior for num_views
                                - added error behavior for tess, geom, xfb, and timer queries
                                - documented new FRAMEBUFFER_ATTACHMENT*OVR variables
                                - clarify that only 2D-array textures are supported
                                - better document new framebuffer completeness conditions
                                - document MAX_VIEWS_OVR and specify minimum value (2)
                                - added issues 18-30
      2     11/15/17  dgkoch    Fix tessellation typos. Fix incorrect reference to compute shaders
                                instead of vertex shaders.
      3     12/13/17  dgkoch    Clarify that compute shaders are orthogonal to multiview
                                framebuffers (Issue 31). Remove gl_ViewID_OVR from compute,
                                tessellation, and geometry shaders.
      4     05/02/18  jhelferty Clarify interop rules for gl_Layer in fragment shader (33), and
                                interop with compatibility profile's Begin/End (34). Add mention
                                of OES_geometry_shader and OES_tessellation_shader to interop.
                                Clarify what happens when transform feedback is paused.
      5     07/25/18  oetuaho   Fix off-by-one issue in baseViewIndex + numViews check.
      6     10/19/18  dgkoch    Add standard boiler plate shader extension language.

