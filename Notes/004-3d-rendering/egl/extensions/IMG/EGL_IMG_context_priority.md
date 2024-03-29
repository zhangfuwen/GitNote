# IMG_context_priority

Name

    IMG_context_priority

Name Strings

    EGL_IMG_context_priority

Contributors

    Ben Bowman, Imagination Techonologies
    Graham Connor, Imagination Techonologies

Contacts

    Ben Bowman, Imagination Technologies (benji 'dot' bowman 'at'
    imgtec 'dot' com)

Status

    Complete

Version

    Version 1.1, 8 September 2009

Number

    EGL Extension #10

Dependencies

    Requires EGL 1.0.

    This extension is written against the wording of the EGL 1.4
    Specification - May 2, 2008 (but may be implemented against earlier
    versions).

Overview

    This extension allows an EGLContext to be created with a priority
    hint. It is possible that an implementation will not honour the
    hint, especially if there are constraints on the number of high
    priority contexts available in the system, or system policy limits
    access to high priority contexts to appropriate system privilege
    level. A query is provided to find the real priority level assigned
    to the context after creation.

New Types

    None

New Procedures and Functions

    None

New Tokens

    New attributes accepted by the <attrib_list> argument of
    eglCreateContext

        EGL_CONTEXT_PRIORITY_LEVEL_IMG          0x3100

    New attribute values accepted in the <attrib_list> argument
    of eglCreateContext:

        EGL_CONTEXT_PRIORITY_HIGH_IMG           0x3101
        EGL_CONTEXT_PRIORITY_MEDIUM_IMG         0x3102
        EGL_CONTEXT_PRIORITY_LOW_IMG            0x3103

Additions to Chapter 3 of the EGL 1.4 Specification (EGL Functions and Errors)

    Modify the list of attributes supported by eglCreateContext in
    section 3.7.1 (Creating Rendering Contexts) on p. 42:

       "<attrib_list> specifies a list of attributes for the context.
        The list has the same structure as described for
        eglChooseConfig. The only attributes that can be specified in
        <attrib_list> are EGL_CONTEXT_CLIENT_VERSION and
        EGL_CONTEXT_PRIORITY_LEVEL_IMG. The EGL_CONTEXT_CLIENT_VERSION
        attribute may only be specified when creating a OpenGL ES
        context (e.g. when the current rendering API is
        EGL_OPENGL_ES_API).

        <attrib_list> may be NULL or empty (first attribute is
        EGL_NONE), in which case attributes assume their default values
        as described below.

        EGL_CONTEXT_CLIENT_VERSION determines which version of an OpenGL
        ES context to create. An attribute value of 1 specifies creation
        of an OpenGL ES 1.x context. An attribute value of 2 specifies
        creation of an OpenGL ES 2.x context. The default value for
        EGL_CONTEXT_CLIENT_VERSION is 1.

        EGL_CONTEXT_PRIORITY_LEVEL_IMG determines the priority level of
        the context to be created. This attribute is a hint, as an
        implementation may not support multiple contexts at some
        priority levels and system policy may limit access to high
        priority contexts to appropriate system privilege level. The
        default value for EGL_CONTEXT_PRIORITY_LEVEL_IMG is
        EGL_CONTEXT_PRIORITY_MEDIUM_IMG."


    Modify the list of attributes supported by eglQueryContext in
    section 3.7.4 (Context Queries) on p. 46:

       "eglQueryContext returns in <value> the value of attribute for
        <ctx>. <attribute> must be set to EGL_CONFIG_ID,
        EGL_CONTEXT_CLIENT_TYPE, EGL_CONTEXT_CLIENT_VERSION,
        EGL_RENDER_BUFFER, or EGL_CONTEXT_PRIORITY_LEVEL_IMG.

        ...

        Querying EGL_CONTEXT_PRIORITY_LEVEL_IMG returns the priority
        this context was actually created with. Note: this may not be
        the same as specified at context creation time, due to
        implementation limits on the number of contexts that can be
        created at a specific priority level in the system."

ISSUES:

    1) Should the context priority be treated as a hint or a requirement

    RESOLVED: The context priority should be a hint. System policy may
    limit high priority contexts to appropriate system privilege level.
    Implementations may have a limit on the number of context supported
    at each priority, and may require all contexts within a process to
    have the same priority level.

    2) Can an application find out what priority a context was assigned?

    RESOLVED: Provide a query to find the assigned priority for a
    context. An application may find that it has a lower (or higher)
    priority than requested (although it probably cannot do much with
    the information).

    3) How many priority levels should be defined?

    RESOLVED: Three seems appropriate, as the highest provides the
    largest GPU timeslice and reduced latency. It might be useful to
    specify a low priority context which has a small timeslice and high
    latency. It is possible that a request for LOW will actually return
    MEDIUM on an implementation that doesn't differentiate between the
    lower two levels.

    4) What should the default priority level be if not specified?

        OPTION 1: HIGH - This allows applications that are unaware of
        this extension to get the highest priority possible.

        OPTIONS 2: MEDIUM - This allows truly high priority applications
        to differentiate themselves from applications which are unaware
        of this extension.

        RESOLVED:
            OPTION 2: MEDIUM - Allow truly high priority applications to
            differentiate themselves.

Revision History
    Version 1.1, 08/09/2009 (Jon Leech) Assign extension number and
        publish in the Registry. Formatting cleanup.
    Version 1.0, 30/04/2009 - Final clean up. Marked issues as resolved,
        take out draft status
    Version 0.3, 22/04/2009 - enums assigned from Khronos registry.
    Version 0.2, 02/04/2009 - feedback from gdc.
    Version 0.1, 31/03/2009 - first draft.
