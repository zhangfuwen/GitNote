# EXT_compositor

Name

  EXT_compositor

Name Strings

  EGL_EXT_compositor

Contributors

  Marc Moody
  Daniel Herring

Contacts
  
  Marc Moody, Boeing Inc., marc dot d dot moody at boeing dot com
  Daniel Herring, Core Avionics & Industrial Inc., daniel dot herring at ch1group dot com

Status

  Complete

Version
  Version 1.0, February 3, 2017

Number

  EGL Extension #116

Dependencies

  Requires EGL 1.0.

  This extension is written against the wording of the EGL 1.4
  Specification - April 6, 2011, but may be implemented against earlier
  versions.

Overview

  This extension allows for the composition of multiple windows within
  a multi-partition EGL system. The extension allows a primary EGLContext and
  window to be created for each display. All other windows are created using
  non-displayable surfaces. A handle to each off-screen window is provided
  to the primary EGLContext to allow the composition of all non-displayable windows
  on a single display.

  For each display, there is one EGLContext which has access to create on screen
  windows, this is call the primary context for this display. All other EGLContexts
  are referred to as secondary contexts.

  This extension requires a multi-partition EGL driver to support asynchronous
  rendering of off screen surfaces.

  Information assurance is provided by preventing context and surface creation by
  unregistered contexts and by preventing the non-primary contexts and surfaces
  from rendering to the display.

New Types

  None

New Procedures and Functions

  EGLBoolean eglCompositorSetContextListEXT (const EGLint *external_ref_ids,
                                                EGLint num_entries);

  EGLBoolean eglCompositorSetContextAttributesEXT (EGLint external_ref_id,
                                       const EGLint *context_attributes,
                                       EGLint num_entries);

  EGLBoolean eglCompositorSetWindowListEXT (EGLint external_ref_id,
                                        const EGLint *external_win_ids,
                                        EGLint num_entries);

  EGLBoolean eglCompositorSetWindowAttributesEXT (EGLint external_win_id,
                                       const EGLint *window_attributes,
                                       EGLint num_entries);

  EGLBoolean eglCompositorBindTexWindowEXT (EGLint external_win_id);

  EGLBoolean eglCompositorSetSizeEXT (EGLint external_win_id,
                                      EGLint width, EGLint height);

  EGLBoolean eglCompositorSwapPolicyEXT (EGLint external_win_id,
                                              EGLint policy);


New Tokens

  New attributes accepted by the <attrib_list> argument of
  eglCreateContext:

    EGL_PRIMARY_COMPOSITOR_CONTEXT_EXT     0x3460

  New attributes accepted by the <attrib_list> argument of
  eglCreateContext and eglCreateWindowSurface:

    EGL_EXTERNAL_REF_ID_EXT                0x3461

  New attributes accepted by the <policy> argument of
  eglCompositorSwapPolicyHint:

    EGL_COMPOSITOR_DROP_NEWEST_FRAME_EXT   0x3462

    EGL_COMPOSITOR_KEEP_NEWEST_FRAME_EXT   0x3463

Modify Section 3.7.1 of the EGL 1.4 Specification, paragraph 2
 (Creating Rendering Contexts) on pg. 43
  From:
  If eglCreateContext succeeds, it initializes the context to the initial state defined
  for the current rendering API, and returns a handle to it. The context can be
  used to render to any compatible EGLSurface.

  To:
  If eglCreateContext succeeds, it initializes the context to the initial state defined
  for the current rendering API, and returns a handle to it. The context can be
  used to render to any compatible off-screen rendering surface (Sections 3.5.2
  and 3.5.4). A secondary context can be used to render to compatible window surfaces
  which have been associated with the context using eglCompositorSetWindowListEXT. A
  non-secondary context can be used to render to any compatible EGLSurface.

Modify Section 3.7.1 of EGL 1.4 Specification, paragraph 5
 (Creating Rendering Contexts) on pg. 43
  From:
  attrib_list specifies a list of attributes for the context. The list has the same
  structure as described for eglChooseConfig. The only attribute that can be specified
  in attrib_list is EGL_CONTEXT_CLIENT_VERSION, and this attribute may only
  be specified when creating a OpenGL ES context (e.g. when the current rendering
  API is EGL_OPENGL_ES_API).

  To:
  attrib_list specifies a list of attributes for the context. The list has the same
  structure as described for eglChooseConfig. The attributes that can be specified
  in attrib_list are EGL_CONTEXT_CLIENT_VERSION, EGL_PRIMARY_COMPOSITOR_CONTEXT_EXT,
  EGL_EXTERNAL_REF_ID_EXT.

Modify Section 3.7.1 of EGL 1.4 Specification, paragraph 7
 (Creating Rendering Contexts) on pg. 43
  From:
  EGL_CONTEXT_CLIENT_VERSION determines which version of an OpenGL
  ES context to create. An attribute value of 1 specifies creation of an OpenGL ES
  1.x context. An attribute value of 2 specifies creation of an OpenGL ES 2.x context.
  The default value for EGL_CONTEXT_CLIENT_VERSION is 1.

  To:
  EGL_CONTEXT_CLIENT_VERSION determines which version of an OpenGL
  ES context to create. An attribute value of 1 specifies creation of an OpenGL ES
  1.x context. An attribute value of 2 specifies creation of an OpenGL ES 2.x context.
  The default value for EGL_CONTEXT_CLIENT_VERSION is 1. EGL_CONTEXT_CLIENT_VERSION
  may only be specified when creating a OpenGL ES context (e.g. when the current
  rendering API is EGL_OPENGL_ES_API).

Additions to Section 3.7.1 of the EGL 1.4 Specification (Creating Rendering Contexts).

  The first call to eglCreateContext with EGL_PRIMARY_COMPOSITOR_CONTEXT_EXT
  set as EGL_TRUE in the <attrib-list> returns an EGLContext handle which will
  act as the primary context for the specified EGLDisplay. This shall be the
  only context on this EGLDisplay which is able to be bound to an on-screen window
  on the EGLDisplay. Subsequent calls to eglCreateContext, for the same display,
  in any address space with EGL_PRIMIARY_COMPOSITOR_CONTEXT_EXT set as EGL_TRUE
  shall return EGL_NO_CONTEXT.
  Subsequent calls, in any address space for the same display, to eglCreateContext
  which do not use EGL_PRIMARY_COMPOSITOR_CONTEXT_EXT should use the the attribute
  EGL_EXTERNAL_REF_ID_EXT and an external reference identifier to create a valid
  EGLContext.
  Subsequent calls, in any address space for the same display, to eglCreateContext
  which do not use EGL_PRIMARY_COMPOSITOR_CONTEXT_EXT and do not use
  EGL_EXTERNAL_REF_ID_EXT shall not create a context and shall return
  EGL_NO_CONTEXT and set the error EGL_BAD_ACCESS.

  EGL_PRIMARY_COMPOSITOR_CONTEXT_EXT notifies EGL that this context is the only
  context allowed to render to a on screen window surface for this display. This
  attribute is followed in the attribute list by either EGL_TRUE or EGL_FALSE.
  EGL_EXTERNAL_REF_ID_EXT is followed by an external reference identifier which
  associates this context with the list of allowed contexts set by
  eglCompositorSetContextListEXT. If the reference identifier has already been
  used to initiate another call to eglCreateContext in any address space, this
  call shall fail and set the error EGL_BAD_ATTRIBUTE. IF the external reference
  identifier is not a valid identifier the error EGL_BAD_ATTRIBUTE shall be set
  and the call shall fail. If this attribute is set and the primary context has
  not yet been created then the error EGL_BAD_MATCH and the call shall fail.

  Only when the attributes specified via attrib_list and those specified for this
  context using eglCompositorSetContextAttributesEXT are compatible will the
  context be successfully created.

Modify the list of parameters supported by eglCreateWindowSurface in section 3.5.1
 (Creating On-Screen Rendering Surfaces) on p. 27:

  An external window identifier (EGLuint external_win_id) is associated with
  each off screen window. EGLNativeWindowType shall be EGLint and the
  external window identifier shall be used as the value for the <win> parameter.
  EGL_EXTERNAL_REF_ID_EXT with the context external reference id shall be in the
  attrib_list. If the external window identifier and the external reference identifier
  do not match a pair set using eglCompositorSetWindowListEXT window creation shall
  fail.

Add the function: 

  EGLBoolean eglCompositorSetContextListEXT(const EGLint *external_ref_ids,
                                            EGLint num_entries);

    This function is called by the primary context to establish the set of
    all allowable secondary contexts by defining the set of external reference
    identifiers. Secondary contexts shall identify their external reference
    identifier by setting the EGL_EXTERNAL_REF_ID_EXT attribute when calling
    eglCreateContext.
    Calls to this function when the primary context for the display is not active,
    shall return EGL_FALSE and do nothing. When this function is successful it
    shall return EGL_TRUE and associate the list of external reference identifiers
    with the active display.
    This function shall only be called once per primary context. Subsequent calls
    shall return EGL_FALSE and do nothing.

    external_ref_ids shall be an array of user generated integers greater than 1.
    num_entries shall specify the number of external_ref_ids.

Add the function:

  EGLBoolean eglCompositorSetContextAttributesEXT (EGLint external_ref_id,
                                     const EGLint *context_attributes,
                                     EGLint num_entries);

    This function is called by the primary context to establish a list of compatible
    context attributes.
    Calls to this function when the primary context for the display is not active
    shall return EGL_FALSE and do nothing. When the function is successful it shall
    return EGL_TRUE and associate the given attributes to the subsequent
    eglCreateContext call using the given external reference id.
    The list of acceptable context_attributes is the same as the list for
    eglCreateContext. Only when this list of attributes is compatible with those
    used during context creation will the secondary context be created.
    This function shall only be called once per secondary context, and must be called
    before the secondary context is able to be created. Subsequent calls shall
    return EGL_FALSE and do nothing.

    num_entries shall specify the number of attributes in the list. This number
     shall prevent accessing memory outside the attribute list, even if EGL_NONE is
     not yet found in the list. If EGL_NONE is found sooner than this number of
     attributes list parsing shall stop.

Add the function:

  EGLBoolean eglCompositorSetWindowListEXT (EGLint external_ref_id,
                                      const EGLint *external_win_ids,
                                      EGLint num_entries);

    This function is called by the primary context to establish the set of
    all allowable windows associated with a specific secondary context by defining
    the set of external window identifiers. Window surface creation within secondary
    contexts use the external window identifier as the native window handle parameter
    of eglCreateWindowSurface and supply an external reference id as a window creation
    attribute.
    Calls to this function when the primary context for the display is not active,
    shall return EGL_FALSE and do nothing. When this function is successful it
    shall return EGL_TRUE and associate the list of external window identifiers
    with the specified secondary context (external_ref_ids).

    external_win_ids shall be an array of user generated integers greater than 1.
     external window identifiers are unique within the primary EGLContext. Any
     external window identifier may be associated with multiple secondary contexts
     however only the first context which calls eglCreateWindowSurface shall
     successfully create a window surface for the given window.
    num_entries shall specify the number of external_ref_ids.

Add the function:

  EGLBoolean eglCompositorSetWindowAttributesEXT (EGLint external_win_id,
                                     const EGLint *window_attributes,
                                     EGLint num_entries);

    This function is called by the primary context to set window specific attributes
    for the specified external window id.

    The list of valid attributes are: EGL_WIDTH, EGL_HEIGHT, EGL_VERTICAL_RESOLUTION,
    EGL_HORIZONRAL_RESOLUTION, EGL_PIXEL_ASPECT_RATIO.
    EGL_WIDTH and EGL_HEIGHT shall set the maximum width and height the secondary
    context is able to create a window surface with.
    EGL_VERTICAL_RESOLUTION, EGL_HORIZONTAL_RESOLUTION, and EGL_PIXEL_ASPECT_RATIO
    shall set the values of these used by the secondary context.
    Calls to this function when the primary context for the display is not active,
    shall return EGL_FALSE and do nothing. When this function is successful it
    shall return EGL_TRUE and associate the list of attributes with the off screen
    window ID (external_win_id).
    This function shall only be called once per window, and must be called before
    the secondary context is able to create the window surface. Subsequent calls
    shall return EGL_FALSE and do nothing.

    num_entries shall specify the number of attributes in the list. This number
    shall prevent accessing memory outside the attribute list, even if EGL_NONE is
    not yet found in the list. If EGL_NONE is found sooner than this number of
    attributes list parsing shall stop.

Add the function:

  EGLBoolean eglCompositorBindTexWindowEXT (EGLint external_win_id);

    This function is similar to eglBindTexImage in that it binds the windows
    color buffers to a texture object which can then be composited on the native
    window by the primary context. This function shall only be callable within
    the primary context while there is a texture object actively bound by the
    native rendering API. The function shall return EGL_TRUE if the window is
    successfully bound to the active texture object. The function shall return
    EGL_FALSE if binding fails, the context is not the primary context, or the
    external_win_id is not a valid external window reference as set using
    eglCompositorSetWindowListEXT.

Add the function:

  EGLBoolean eglCompositorSetSizeEXT(EGLint external_win_id,
                                         EGLint width, EGLint height);

    This function shall be called by the primary context to set the width and
    height of the window. This function returns EGL_FALSE when called by a context
    other than the primary context. This function will also return EGL_FALSE if
    the new width and height are larger than the maximums set by
    eglCompositorSetWindowAttributesEXT. Upon successful window resizing the function
    shall return EGL_TRUE.
    Secondary contexts may use EGL to query the windows width and height at runtime
    to detect window resize events. The window buffer resizing shall be applied
    to the newly active buffer after the secondary context calls eglSwapBuffers.
    This will leave an average of 2 buffers which will have to be scaled by the
    native rendering API to the new resolution.

Add the function:

  EGLBoolean eglCompositorSwapPolicyEXT(EGLint external_win_id,
                                             EGLint policy);

    This function shall be called by the primary context to specify the handling
    of buffer swaps of the context specified by external_ref_id.

    When the policy is set to EGL_COMPOSITOR_DROP_NEWEST_FRAME_EXT, and the
    secondary context completes a frame by calling eglSwapbuffers, and the
    primary context is actively reading from the front buffer associated with
    the external window ID, then the just completed frame is dropped, no buffer
    swap occurs, and eglSwapBuffers will return EGL_TRUE.
    If EGL_COMPOSITOR_KEEP_NEWEST_FRAME_EXT is specified and the primary context
    is actively reading from the front buffer associated with the external window
    ID then the secondary context's call to eglSwapBuffers will return EGL_FALSE.
    eglSwapBuffers will continue to return EGL_FALSE as long as the primary context
    is actively reading from the front buffer. Once the primary context finishes
    reading from the front buffer the next call to eglSwapBuffers will
    return EGL_TRUE. It is up to the application to decide if it will wait until
    eglSwapBuffers returns EGL_TRUE before rendering again. If the secondary
    context continues to render after eglSwapBuffers returns EGL_FALSE then it will
    be as if the swap policy was EGL_COMPOSITOR_DROP_NEWEST_FRAME_EXT.


Revision History
  Version 0.1, 30/08/2014 - first draft.
  Version 0.2, 24/09/2014  second draft.
  Version 0.3, 21/12/2016
     Khronos vendor extension clean-up.
  Version 0.4, 05/01/2016
   - Change from context and display limitation to context and window limitations
   - Added additional functions related to external window ids contexts.
  Version 1.0, 03/02/2017
   - Fixed misc issues related to parameter names and types. 

Question: How are external reference and window IDs assigned and managed? Are they
          reserved somehow within a new EGL C header file?
Answer:   External IDs are assigned from the software component with the primary
          context during an EGL initialization stage using eglCompositorSet*ListEXT.
          The secondary contexts are provided the appropriate external reference ID
          during end system integration. It is recommended that these values be able
          to be set via a configuration parameter to the application, to ease 
          integration of multiple programs.

Question: What happens when the secondary contexts render faster than the
          primary context?
Answer:   This is based on the swap policy set by the compositor, and the rate
          difference between the render and the compositor. Secondary contexts
          will be notified by EGL_FALSE being returned if the windows front buffer
          is currently being read by the compositor. However, if the compositor is
          not currently reading the front buffer the swap will succeed and the
          secondary contexts latest rendered frame will now be in the front buffer
          to be composited.

Question: What happens when the primary context renders faster than the
          secondary context(s)?
Answer:   The contents of the windows front buffer will be repeated.

Question: Does this infer any Z-ordering for the off-screen surfaces?
Answer:   No, the Z-order is applied by the compositor when doing the final on
          screen rendering of the off-screen surfaces. Or may be set if a Z-ordering
          extension is applied to the attribute list of the window using
          eglCompositorSetWindowAttributesEXT

Comments: 

The driver should also have to have enough protection in it to prevent a random
software component from passing random buffer handles to the driver to prevent
access to other software components.

The driver may force use of double buffering event if the window surface was created
using an EGL_RENDER_BUFFER attribute set to EGL_SINGLE_BUFFER.

If EGL_PRIMARY_COMPOSITOR_CONTEXT_EXT is not used to create a context for a specified
display then EGL will act as though this extension is not enabled.
