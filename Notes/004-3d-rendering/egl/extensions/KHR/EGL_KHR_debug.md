# KHR_debug

Name

    KHR_debug

Name Strings

    EGL_KHR_debug

Contributors

    Jeff Vigil, Qualcomm
    Brian Ellis, Qualcomm
    (Original contributors of Gl_KHR_debug extension for OpenGL/GL_ES)
    Mark Callow, HI
    John Leech, Khronos
    Ray Smith, ARM
    Prabindh Sundareson, Texas Instruments
    James Jones, NVIDIA
    Jesse Hall, Google

Contact

    Jeff Vigil (jvigil 'at' qualcomm.com)

Notice

    Copyright (c) 2012-2015 The Khronos Group Inc. Copyright terms at
        http://www.khronos.org/registry/speccopyright.html

Status

    Complete. Approved by the EGL Working Group on 2015/04/24.
    Approved by the Khronos Board of Promoters on 2015/06/26.

Version

    Version 18, Modified Date: September 28, 2016

Number

    EGL Extension #92

Extension Type

    EGL client extension

Dependencies

    Applicable to any version of EGL 1.x, but written in relationship
    to EGL 1.5.

Overview

    This extension allows EGL to notify applications when various events
    occur that may be useful during application development and debugging.

    These events are represented in the form of debug messages with a
    human-readable string representation. Examples of debug events include
    errors due to incorrect use of the EGL API, warnings of undefined behavior,
    and performance warnings.

    The "type" of the message roughly identifies the nature of the event that
    caused the message. Examples include input errors, performance
    information, or warnings about undefined behavior.

    Messages are communicated to the application through an application-
    defined callback function that is called by the EGL implementation on
    each debug message. The motivation for the callback routine is to free
    application developers from actively having to query whether an EGL error,
    or any other debuggable event has happened after each call to a EGL
    function. With a callback, developers can keep their code free of debug
    checks, set breakpoints in the callback function, and only have to react
    to messages as they occur. The callback also offers much more information
    than just an error code.

    To control the volume of debug output, types of messages can be enabled or
    disabled. The mechanism is controlled by attributes passed to EGL. The
    state of the message type control can be queried.

    Debug output can be enabled and disabled by changing the callback function.
    NULL will disable the feature while a valid function pointer will enable
    it.

    Finally, this extension defines a mechanism for EGL applications to
    label their objects (contexts, surfaces, syncs, etc.) with a pointer
    to an application provided structure. This pointer can be a descriptive
    string, identifier or pointer to a structure. This enables the application
    to associate the EGL object with application information. EGL will not
    interpret this pointer as a string or any other meaning - just attach to
    the object and pass back in the callback when that object is the primary
    object of an event.

IP Status

    No known IP claims.

New Procedures and Functions

    EGLint eglDebugMessageControlKHR(
            EGLDEBUGPROCKHR callback,
            const EGLAttrib* attrib_list);

    EGLBoolean eglQueryDebugKHR(
            EGLint attribute,
            EGLAttrib* value);

    EGLInt eglLabelObjectKHR(
            EGLDisplay display,
            EGLenum objectType,
            EGLObjectKHR object,
            EGLLabelKHR label);

New Types

    A general type to identify EGL objects, such as EGLSurface or EGLContext.

        typedef void* EGLObjectKHR;

    A label is a string, ID or pointer to a structure that the application
    can attach to an EGL object.

        typedef void* EGLLabelKHR;

    The callback function that applications can define, and is accepted by
    eglDebugMessageControlKHR, is defined as:

        typedef void (APIENTRY *EGLDEBUGPROCKHR)(
                EGLenum error,
                const char *command,
                EGLint messageType,
                EGLLabelKHR threadLabel,
                EGLLabelKHR objectLabel,
                const char* message);

New Tokens

    Tokens accepted by the <objectType> parameter of function
    eglLabelObjectKHR:

        EGL_OBJECT_THREAD_KHR                            0x33B0
        EGL_OBJECT_DISPLAY_KHR                           0x33B1
        EGL_OBJECT_CONTEXT_KHR                           0x33B2
        EGL_OBJECT_SURFACE_KHR                           0x33B3
        EGL_OBJECT_IMAGE_KHR                             0x33B4
        EGL_OBJECT_SYNC_KHR                              0x33B5
        EGL_OBJECT_STREAM_KHR                            0x33B6

    Tokens provided by the <messageType> parameter of EGLDEBUGPROCKHR
    or the attributes input to eglControlDebugMessageKHR or attribute
    of eglQueryDebugKHR:

        EGL_DEBUG_MSG_CRITICAL_KHR                       0x33B9
        EGL_DEBUG_MSG_ERROR_KHR                          0x33BA
        EGL_DEBUG_MSG_WARN_KHR                           0x33BB
        EGL_DEBUG_MSG_INFO_KHR                           0x33BC

    Tokens provided by the input attribute to eglQueryDebugKHR:

        EGL_DEBUG_CALLBACK_KHR                           0x33B8

Additions to Chapter 3 of the EGL 1.5 Specification
(EGL Functions and Errors)

    Add new Section 3.13:

    "3.13 - Debug Output

    Application developers can obtain more information from EGL runtime in
    the form of debug output.  This information can include details about EGL
    errors, undefined behavior, implementation-dependent performance warnings,
    or other useful hints.

    This information is communicated through a stream of debug messages that
    are generated as EGL commands are executed.  The application can
    receive these messages through a callback routine.

    Controls are provided for disabling classes of messages that the
    application does not care about.

    Debug output functionality is controlled with:

        EGLint eglDebugMessageControlKHR(
                EGLDEBUGPROCKHR callback,
                const EGLAttrib* attrib_list);

    If the <callback> parameter is NULL, then no messages are sent to the
    callback function and the debug message generation is disabled. If the
    <callback> parameter is a pointer to a valid callback function, as defined
    by EGLDEBUGPROCKHR, then messages will be sent to that callback function.

    The attribute list <attrib_list> contains a set of message type enums,
    and each has a value of EGL_TRUE to enable that class of messages,
    or value EGL_FALSE to disable that class of message.

    If the <attrib_list> contains an unknown attribute or value the function
    will return a EGL_BAD_ATTRIBUTE error.
    
    If there is no error, then the function will set the updated callback,
    set the updated message types and return EGL_SUCCESS.

    The messages types, their purpose and initial states are given in
    table 13.1 below. The parameter <attrib_list> needs only contain the
    attributes to change; an application can call eglDebugMessageControl more
    than once with a valid callback, and change the message type states as
    desired.

    When the callback is set to NULL; the attributes are reset to their
    default values.

    Debug Output Message Type     Informs about                   Initial/Default state
    -------------------------     -------------                   ---------------------
    EGL_DEBUG_MSG_CRITICAL_KHR    Internal EGL driver failures    ENABLED
                                  i.e. EGL_BAD_ALLOC,
                                  EGL_CONTEXT_LOST

    EGL_DEBUG_MSG_ERROR_KHR       Input and bad match errors      ENABLED
                                  i.e. EGL_BAD_CONTEXT,
                                  EGL_BAD_PARAMETER...

    EGL_DEBUG_MSG_WARN_KHR        Warnings, code is EGL_SUCCESS,  DISABLED
                                  but message indicates
                                  deprecated or inefficient
                                  operation.

    EGL_DEBUG_MSG_INFO_KHR        Verbose operation               DISABLED
                                  Messages such as object
                                  creation and destruction
                                  or change in state.

    ---------------------------------------------------------------------------
    Table 13.1: Types of debug output messages. Each debug message is associated
    with one of these types that describes the nature or class of the message.

    3.13.1 - Debug Message Callback

    Applications must provide a callback function for receiving debug messages
    of the following type:

        typedef void (APIENTRY *EGLDEBUGPROCKHR)(
                EGLenum error,
                const char *command,
                EGLint messageType,
                EGLLabelKHR threadLabel,
                EGLLabelKHR objectLabel,
                const char* message);

    The function's prototype must follow the type definition of EGLDEBUGPROCKHR.
    Only one debug callback can be in-use for the application, and
    further calls overwrite the previous callback. Specifying NULL as the
    value of <callback> clears the current callback and disables message
    output.

    The callback will receive the following in its parameters:

        <error> will contain an EGL error code, or EGL_SUCCESS, as applicable.

        <command> will contain a pointer to a string. Example "eglBindApi".

        <messageType> will contain one of the debug message types listed in
        table 13.1.

        <threadLabel> will contain the label attached to the current thread.
        The <threadLabel> will be NULL if not set by the application. If the
        message is from an internal thread, the label will be NULL.

        <objectLabel> will contain the label attached to the primary object
        of the message; Labels will be NULL if not set by the application.
        The primary object should be the object the function operates on, see
        table 13.2 which provides the recommended mapping between functions and
        their primary object. This <objectLabel> may be NULL even though the
        application labeled the object. This is because it is possible an error
        was raised while executing the command before the primary object was
        validated, therefore its label cannot be included in the callback.

        <message> will contain a platform specific debug string message;
        This string should provide added information to the application
        developer regarding the condition that generated the message.
        The format of a message is implementation-defined, although it should
        represent a concise description of the event that caused the message
        to be generated. Message strings can be NULL and should not be assumed
        otherwise.

    EGL Command                             Primary object
    -------------------------               -------------

      eglBindAPI                             thread
      eglBindTexImage                        surface
      eglChooseConfig                        display
      eglClientWaitSync                      sync
      eglCopyBuffers                         surface
      eglCreateContext                       display
      eglCreateImage                         display
      eglCreatePbufferFromClientBuffer       display
      eglCreatePbufferSurface                display
      eglCreatePixmapSurface                 display
      eglCreatePlatformWindowSurface         display
      eglCreatePlatformPixmapSurface         display
      eglCreateSync                          display
      eglCreateWindowSurface                 display
      eglDestroyContext                      context
      eglDestroyImage                        image
      eglDestroySurface                      surface
      eglDestroySync                         sync
      eglGetConfigAttrib                     display
      eglGetConfigs                          display
      eglGetCurrentContext                   context
      eglGetCurrentDisplay                   display
      eglGetCurrentSurface                   surface
      eglGetDisplay                          thread
      eglGetError                            thread
      eglGetPlatformDisplay                  thread
      eglGetSyncAttrib                       sync
      eglInitialize                          display
      eglMakeCurrent                         context
      eglQueryAPI                            context
      eglQueryContext                        context
      eglQueryString                         display
      eglQuerySurface                        surface
      eglReleaseTexImage                     surface
      eglReleaseThread                       thread
      eglSurfaceAttrib                       surface
      eglSwapBuffers                         surface
      eglSwapInterval                        surface
      eglTerminate                           display
      eglWaitClient                          context
      eglWaitGL                              context
      eglWaitNative                          thread
      eglWaitSync                            sync
      eglDebugMessageControlKHR              -none-
      eglQueryDebugKHR                       -none-
      eglLabelObjectKHR                      labeled object

    ---------------------------------------------------------------------------
    Table 13.2: Recommendation of primary object in a callback as result
    of various EGL commands.

    If the application has specified a <callback> function for receiving debug
    output, the implementation will call that function whenever any enabled
    message is generated. A message must be posted for every error since
    debug messages can be used as an alternative to eglGetError() for error
    detection and handling. Specifying a callback function does not affect the
    behavior of eglGetError; errors are reported through both mechanisms.

    Applications that specify a callback function must be aware of certain
    special conditions when executing code inside a callback when it is
    called by EGL. The memory for <message> is read-only, owned and managed
    by EGL, and should only be considered valid for the duration of the
    function call. Likewise the <command> string is read-only EGL managed
    memory and should be considered valid only for the duration of the
    callback.

    Setting the label for EGL objects is optional and only intended for
    applications to correlate application structures with EGL objects.
    All object labels are initially NULL.

    The behavior of calling any EGL operation, its client APIs, or window system
    functions from within the callback function is undefined and may lead
    to program termination. It should not be considered reentrant.

    Only one debug callback may be registered at a time; registering a new
    callback will replace the previous callback. The callback is used by any
    thread that calls EGL, so if the application calls into EGL concurrently
    from multiple threads it must ensure the callback is thread-safe.

    EGL may employ internal threads to execute EGL commands. These threads can
    post debug messages to the callback function. The labels for these
    internal threads will be NULL.

    3.13.2 Debug Labels:

    Debug labels provide a method for annotating any object (context, surface,
    sync, etc.) with an application provided label. These labels may then be
    used by the debug output or an external tool such as a debugger or profiler
    to describe labeled objects.

     The command

        EGLint eglLabelObjectKHR(
            EGLDisplay display,
            EGLenum objectType,
            EGLObjectKHR object,
            EGLLabelKHR label);

    enables the application to attach a label to a specified object.
    The <display>, <objectType>, and <object> identify the object to be
    labeled.

    The <label> contains a pointer sized variable to attach to the
    object. This label can be a integer identifier, string or pointer to a
    application defined structure. EGL will not interpret this value;
    it will merely provide it when the object is involved in a callback
    message. The label for any object will initially be NULL until set by
    the application.

    An EGL_BAD_PARAMETER error is returned by eglLabelObjectKHR if <objectType>
    doesn't match one of the object type enums. An EGL_BAD_PARAMETER is also
    returned if the <objectType> is not a supported type; such as no support
    for streams.

    An EGL_BAD_PARAMETER error is returned by eglLabelObjectKHR if <object> is
    invalid, unknown, NULL, or is not an object created with
    EGLDisplay <display>.

    When the <objectType> is EGL_OBJECT_THREAD_KHR, the <object> parameter
    will be ignored by EGL. The thread is implicitly the active thread. It is
    recommended that the application pass a NULL for the <object> parameter in
    this case.

    When the <objectType> is EGL_OBJECT_DISPLAY_KHR, the <object> parameter
    must be the same as the <display> parameter - the Display to label. If
    these do not match, in this case, a EGL_BAD_PARAMETER is generated.

    The <display> does not need to be initialized for <objectType>
    EGL_OBJECT_THREAD_KHR, or EGL_OBJECT_DISPLAY_KHR; However for all other
    types it must be initialized in order to validate the <object> for
    attaching a label.

    If there is no error, then the function will set the label and return
    EGL_SUCCESS.

    3.13.3 Debug Queries:

     The command

    EGLBoolean eglQueryDebugKHR(
            EGLint attribute,
            EGLAttrib* value);

    enables the application to query the current value for the debug
    attributes. On success the function returns EGL_TRUE.

    If <attribute> is a message type enum, the value returned will
    be either EGL_TRUE or EGL_FALSE to indicate whether the specified types of
    messages are enabled or disabled respectively.

    Querying for attribute EGL_DEBUG_CALLBACK_KHR will return the current
    callback pointer. This feature is intended to enable layering of the
    callback with helper libraries.

    Querying for an unknown attribute will result in an EGL_BAD_ATTRIBUTE error
    and a return of EGL_FALSE.

Usage Examples

    This example shows starting debug messaging and attaching string labels to
    newly created objects.

    void MyCallBack(EGLenum error,
                    const char *command,
                    EGLint messageType,
                    EGLLabelKHR threadLabel,
                    EGLLabelKHR objectLabel,
                    const char* message)
    {
        printf("Error: %x, With command %s, Type: %d,"
            "Thread: %s, Object: %s, Message: %s.",
            error, command, messageType, threadLabel, objectLabel, message);
    }

    EGLint result;

    // DEBUG_MSG_ERROR and CRITICAL are enabled by default
    EGLAttrib debugAttribs = {EGL_DEBUG_MSG_WARN_KHR, EGL_TRUE, EGL_NONE};
    // Start up debug messaging:
    result = eglDebugMessageControl(MyCallBack, debugAttribs);

    // Label for the rendering thread.
    EGLLabelKHR renderThreadLabel = (EGLLabelKHR)"Render thread";
    result = eglLabelObject(NULL, EGL_OBJECT_THREAD_KHR, NULL, renderThreadLabel);

    EGLDisplay dpy = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    EGLLabelKHR myDisplay = (EGLLabelKHR)"Default display";
    result = eglLabelObject(dpy, EGL_OBJECT_DISPLAY_KHR, dpy, myDisplay);

    eglInitialize(dpy);

    EGLLabelKHR renderContextLabel = (EGLLabelKHR)"Render context";
    EGLContext ctx = eglCreateContext(dpy, config, NULL, contextAttribs);
    result = eglLabelObject(dpy, EGL_OBJECT_CONTEXT_KHR, ctx, renderContextLabel);


Issues

  1. Why not use GL_KHR_debug?

     RESOLVED: Most EGL use and object creation happens before creating a
     GL context. And since EGL operations are thread related - the debug
     messages should be too.

  2. Is the callback expected only to be called from the thread which it's
     registered?

     RESOLVED: In most cases when an application thread calls an EGL function,
     it is expected that EGL upon detecting an error will callback using that
     application thread. However, EGL may have internal helper threads that
     execute operations. These threads can callback but will have no
     threadLabel. It is the responsibility of EGL to ensure that if these
     threads are blocked in the application's callback by a breakpoint; that
     EGL does not fail. Internal threads are an implementation detail and
     are not required.


Revision History

    Revision 18, 2016-07-28 (Jeff Vigil)
      - Clarify return values/error codes.

    Revision 17, 2015-09-23 (Jeff Vigil)
      - Correct type name to "EGLDEBUGPROCKHR". Updated example code.

    Revision 16, 2015-04-15 (Jeff Vigil)
      - Clarified that <objectLabel> maybe NULL in the callback, if an error
      is raised before the primary object handle is validated.

    Revision 15, 2015-03-30 (Jeff Vigil)
      - Further details to labeling of EGL_OBJECT_DISPLAY_KHR.

    Revision 14, 2015-03-27 (Jeff Vigil)
      - Further clarification of returns and errors. Add further details to
      labeling of EGL_OBJECT_THREAD_KHR and EGL_OBJECT_DISPLAY_KHR.

    Revision 13, 2015-03-26 (Jeff Vigil)
      - Clarified returns and errors.

    Revision 12, 2015-03-24 (Jeff Vigil)
      - Improve readability. Add assigned enum values.

    Revision 11, 2015-03-02 (Jeff Vigil)
      - Clarify text regarding parameter attribute_list and its persistence.

    Revision 10, 2015-02-25 (Jeff Vigil)
      - Clarify text regarding callback blocking.
      - The implementation must provide errors and success in callbacks so
      that the callback replaces the use of eglGetError.
      - <command> strings are read-only EGL memory.
      - Specify default values for attributes.
      - Fix typos.

    Revision 9, 2015-02-03 (Jeff Vigil)
      - Updated contributors.
      - Add extension type.
      - Add "KHR" to token and function names.
      - Fix typos.
      - Add query to get current callback pointer.

    Revision 8, 2014-12-03 (Jeff Vigil)
      - Add table containing recommendation for primary object in the callback.

    Revision 7, 2014-10-21 (Jeff Vigil)
      - Remove configs as a label-able object.
      - Remove redundant text.
      - Simplify to one callback per process, not per thread.

    Revision 6, 2014-10-17 (Jeff Vigil)
      - Add issues.
      - Address internal EGL threads posting messages.

    Revision 5, 2014-05-27 (Jeff Vigil)
      - Add missing text for eglQueryDebug.
      - Clarify threading model.

    Revision 4, 2014-04-14 (Jeff Vigil)
      - Fix due to feedback from EGL WG face-to-face conference.

    Revision 3, 2014-04-10 (Jeff Vigil)
      - Refinements.

    Revision 2, 2014-02-21 (Jeff Vigil)
      - Simplify API.

    Revision 1, 2013-09-08 (Jeff Vigil)
      - Work in progress for F2F, Based on GL_KHR_debug, replace GL with EGL
      and remove GL spec specific text.
