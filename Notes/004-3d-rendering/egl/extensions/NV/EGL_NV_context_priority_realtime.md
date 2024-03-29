# NV_context_priority_realtime

Name

    NV_context_priority_realtime

Name Strings

    EGL_NV_context_priority_realtime

Contributors

    Sandeep Shinde
    Kirill Artamonov
    Sami Kiminki
    Donghan Ryu
    Daniel Koch
    Mark Kilgard

Contacts

    Sandeep Shinde, NVIDIA (sashinde 'at' nvidia 'dot' com)

Status

    Complete

Version

    Version 4 - November 21, 2017

Number

    EGL Extension #124

Dependencies

    Requires EGL 1.0.

    Requires EGL_IMG_context_priority

    This extension is written against the wording of the EGL 1.5
    Specification - August 27, 2014 (but may be implemented against earlier
    versions).

Overview

    This extension allows an EGLContext to be created with one extra
    priority level in addition to three priority levels that are part of
    EGL_IMG_context_priority extension.

    This new level has extra privileges that are not available to other three
    levels. Some of the privileges may include:
    - Allow realtime priority to only few contexts
    - Allow realtime priority only to trusted applications
    - Make sure realtime priority contexts are executed immediately
    - Preempt any current context running on GPU on submission of
      commands for realtime context

    The mechanism for determining which EGL context is allowed to use this
    priority level is platform dependent.

New Types

    None

New Procedures and Functions

    None

New Tokens

    New attribute value accepted for the EGL_CONTEXT_PRIORITY_LEVEL_IMG
    attribute in the <attrib_list> argument of eglCreateContext:

        EGL_CONTEXT_PRIORITY_REALTIME_NV         0x3357

Additions to Chapter 3 of the EGL 1.5 Specification (EGL Functions and Errors)

    Add a NEW section "3.7.1.7 Context Priority" to specify the context
    priority attribute for EGL_IMG_context_priority and values:

    "3.7.1.7 Context Priority

    The attribute EGL_CONTEXT_PRIORITY_LEVEL_IMG specifies a context
    priority hint for a context supporting context priority.  This
    attribute's value may be one of EGL_CONTEXT_PRIORITY_HIGH_IMG,
    EGL_CONTEXT_PRIORITY_MEDIUM_IMG, EGL_CONTEXT_PRIORITY_LOW_IMG,
    or EGL_CONTEXT_PRIORITY_REALTIME_NV.  The default value for
    EGL_CONTEXT_PRIORITY_LEVEL_IMG is EGL_CONTEXT_PRIORITY_MEDIUM_IMG.

    This attribute is a hint, as an implementation may not support
    multiple contexts at some priority levels and system policy may limit
    access to high priority contexts to appropriate system privilege
    level.

    The value EGL_CONTEXT_PRIORITY_REALTIME_NV requests the created
    context run at the highest possible priority and be capable of
    preempting the current executing context when commands are flushed
    by such a realtime context.

    This attribute is supported only for OpenGL and OpenGL ES contexts."

    Within section 3.7.4 "Context Queries" amend the eglQueryContext
    discussion as follows:

    Change the sentence describing the attribute parameter to include
    EGL_CONTEXT_PRIORITY_LEVEL_IMG so it reads:

    "attribute must be set to EGL_CONFIG_ID, EGL_CONTEXT_CLIENT_TYPE,
    EGL_CONTEXT_CLIENT_VERSION, EGL_RENDER_BUFFER, or
     EGL_CONTEXT_PRIORITY_LEVEL_IMG."

    After the discussion of "Querying EGL_RENDER_BUFFER", add:

    "Querying EGL_CONTEXT_PRIORITY_LEVEL_IMG returns the priority this
    context was actually created with. Note: this may not be the same
    as specified at context creation time, due to implementation limits
    on the number of contexts that can be created at a specific priority
    level in the system."

Issues

    1)  The EGL_IMG_context_priority extension is written to amend the EGL
        1.4 specification.  Should this extension amend EGL 1.5 or 1.4?

        RESOLVED:  EGL 1.5 because it is newer and better organized to
        extend context attributes.

        EGL 1.5 rewrites 3.7.1 "Creating Rendering Contexts" to have subsections
        for different context attributes.  This extension adds a new such section
        that includes the EGL_IMG_context_priority attribute and values too.

    2)  Is context priority hint supported for both OpenGL and OpenGL ES contexts?

        RESOLVED:  Yes.

    3)  What is the intended application of the realtime priority level?

        RESOLVED:  One anticipated application is the system compositor
        for a Head Mounted Display (HMD) requires realtime recomposition
        for time-warping.

    4)  What action causes a context with realtime priority to preempt
        other contexts?

        RESOLVED:  Preemption by a context with realtime priority should
        occur when there are pending rendering commands and an implicit or
        explicit flush (i.e. glFlush or glFinish) occurs.

    5)  What does "trusted" or "appropriate system privilege level"
        mean in practice for a Linux-based operating system such as Android?

        RESOLVED: Trusted means an application that has higher privileges
        than other apps such as having CAP_SYS_NICE capability. On Android
        such applications have to be registered in advance with the OS;
        unpriviledged third party app cannot acquire this capability.

        This restriction exists so arbitrary applications do not starve or
        otherwise compromise the interactivity of the system overall.

    6)  In practice how many realtime priority contexts can exist in a system to
        get best performance?

        RESOLVED: Only one realtime priority context should be active at a given
        moment to get best performance.

    7)  Can a context created with a realtime priority hint that is
        in fact given a realtime priority, subsequently find that realtime
        priority revoked and, if revoked, can it be restored?

        RESOLVED: No, once a context is created with specific priority level, the
        priority will not change for lifetime of the context. This means there will
        not be revoking or restoring of realtime priority to already created context.

    8)  The attrib_list for eglCreateContext could list the attribute
        EGL_CONTEXT_PRIORITY_LEVEL_IMG multiple times with different valid values.
        What happens in this case?

        RESOLVED: Behavior is undefined in this case.

        NVIDIA's EGL implementation handles such case by using the last (valid) attribute
        value listed in the attrib_list array as the effective attribute value for
        creating the context.

        The EGL specification is unfortunately silent on this issue.


Revision History
    Version 1,  2016/11/23 (Sandeep Shinde)
        - Initial version
    Version 2,  2017/10/13 (Mark Kilgard)
        - Complete and convert to NV extension
    Version 3, 2017/10/31 (Sandeep Shinde)
        - Few minor corrections. Issue 6 resolved.
    Version 4, 2017/11/21 (Sandeep Shinde)
        - Update enum value and add extension number
