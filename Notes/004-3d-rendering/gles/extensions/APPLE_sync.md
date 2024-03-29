# APPLE_sync

Name

    APPLE_sync

Name Strings

    GL_APPLE_sync

Contributors

    Contributors to ARB_sync desktop OpenGL extension from which this 
    extension borrows heavily
    
Contact

    Benj Lipchak (lipchak 'at' apple 'dot' com)

Status

    Complete

Version

    Last Modified Date: July 10, 2012
    Author Revision: 3

Number

    OpenGL ES Extension #124

Dependencies

    OpenGL ES 1.1 or OpenGL ES 2.0 is required.
    
    This specification is written against the OpenGL ES 2.0.25 specification.
    
    EXT_debug_label affects the definition of this extension.

Overview

    This extension introduces the concept of "sync objects". Sync
    objects are a synchronization primitive - a representation of events
    whose completion status can be tested or waited upon. One specific
    type of sync object, the "fence sync object", is supported in this
    extension, and additional types can easily be added in the future.

    Fence sync objects have corresponding fences, which are inserted
    into the OpenGL command stream at the time the sync object is
    created. A sync object can be queried for a given condition. The
    only condition supported for fence sync objects is completion of the
    corresponding fence command. Fence completion allows applications to
    request a partial Finish, wherein all commands prior to the fence
    will be forced to complete before control is returned to the calling
    process.

    These new mechanisms allow for synchronization between the host CPU
    and the GPU, which may be accessing the same resources (typically
    memory), as well as between multiple GL contexts bound to multiple
    threads in the host CPU.

New Types

    (Implementer's Note: GLint64 and GLuint64 are defined as appropriate
    for an ISO C 99 compiler. Other language bindings, or non-ISO
    compilers, may need to use a different approach).

    #include <inttypes.h>
    typedef int64_t GLint64;
    typedef uint64_t GLuint64;
    typedef struct __GLsync *GLsync;

New Procedures and Functions

    sync FenceSyncAPPLE(enum condition, bitfield flags);
    boolean IsSyncAPPLE(sync sync);
    void DeleteSyncAPPLE(sync sync);

    enum ClientWaitSyncAPPLE(sync sync, bitfield flags, uint64 timeout);
    void WaitSyncAPPLE(sync sync, bitfield flags, uint64 timeout);

    void GetInteger64vAPPLE(enum pname, int64 *params);
    void GetSyncivAPPLE(sync sync, enum pname, sizei bufSize, sizei *length,
        int *values);

New Tokens

    Accepted as the <pname> parameter of GetInteger64vAPPLE:

        MAX_SERVER_WAIT_TIMEOUT_APPLE        0x9111

    Accepted as the <pname> parameter of GetSyncivAPPLE:

        OBJECT_TYPE_APPLE                    0x9112
        SYNC_CONDITION_APPLE                 0x9113
        SYNC_STATUS_APPLE                    0x9114
        SYNC_FLAGS_APPLE                     0x9115

    Returned in <values> for GetSynciv <pname> OBJECT_TYPE_APPLE:

        SYNC_FENCE_APPLE                     0x9116

    Returned in <values> for GetSyncivAPPLE <pname> SYNC_CONDITION_APPLE:

        SYNC_GPU_COMMANDS_COMPLETE_APPLE     0x9117

    Returned in <values> for GetSyncivAPPLE <pname> SYNC_STATUS_APPLE:

        UNSIGNALED_APPLE                     0x9118
        SIGNALED_APPLE                       0x9119

    Accepted in the <flags> parameter of ClientWaitSyncAPPLE:

        SYNC_FLUSH_COMMANDS_BIT_APPLE        0x00000001

    Accepted in the <timeout> parameter of WaitSyncAPPLE:

        TIMEOUT_IGNORED_APPLE                0xFFFFFFFFFFFFFFFFull

    Returned by ClientWaitSyncAPPLE:

        ALREADY_SIGNALED_APPLE               0x911A
        TIMEOUT_EXPIRED_APPLE                0x911B
        CONDITION_SATISFIED_APPLE            0x911C
        WAIT_FAILED_APPLE                    0x911D

    Accepted by the <type> parameter of LabelObjectEXT and 
    GetObjectLabelEXT:

        SYNC_OBJECT_APPLE                    0x8A53


Additions to Chapter 2 of the OpenGL ES 2.0 Specification (OpenGL ES Operation)

    Add to Table 2.2, GL data types:

   "GL Type  Minimum    Description
             Bit Width
    -------  ---------  ----------------------------------------------
    int64    64         Signed 2's complement binary integer
    uint64   64         Unsigned binary integer
    sync     <ptrbits>  Sync object handle (see section 5.2)"


Additions to Chapter 5 of the OpenGL ES 2.0 Specification (Special Functions)

    Insert a new section following "Flush and Finish" (Section 5.1)
    describing sync objects and fence operation. Renumber existing
    section 5.2 "Hints" and all following 5.* sections.

   "5.2 Sync Objects and Fences
    ---------------------------

    Sync objects act as a <synchronization primitive> - a representation
    of events whose completion status can be tested or waited upon. Sync
    objects may be used for synchronization with operations occuring in
    the GL state machine or in the graphics pipeline, and for
    synchronizing between multiple graphics contexts, among other
    purposes.

    Sync objects have a status value with two possible states:
    <signaled> and <unsignaled>. Events are associated with a sync
    object. When a sync object is created, its status is set to
    unsignaled. When the associated event occurs, the sync object is
    signaled (its status is set to signaled). The GL may be asked to 
    wait for a sync object to become signaled.

    Initially, only one specific type of sync object is defined: the
    fence sync object, whose associated event is triggered by a fence
    command placed in the GL command stream. Fence sync objects are used
    to wait for partial completion of the GL command stream, as a more
    flexible form of Finish.

    The command

    sync FenceSyncAPPLE(enum condition, bitfield flags);

    creates a new fence sync object, inserts a fence command in the GL
    command stream and associates it with that sync object, and returns
    a non-zero name corresponding to the sync object.

    When the specified <condition> of the sync object is satisfied by
    the fence command, the sync object is signaled by the GL, causing
    any ClientWaitSyncAPPLE or WaitSyncAPPLE commands (see below) blocking 
    on <sync> to <unblock>. No other state is affected by FenceSyncAPPLE 
    or by execution of the associated fence command.

    <condition> must be SYNC_GPU_COMMANDS_COMPLETE_APPLE. This condition 
    is satisfied by completion of the fence command corresponding to the
    sync object and all preceding commands in the same command stream.
    The sync object will not be signaled until all effects from these
    commands on GL client and server state and the framebuffer are fully
    realized. Note that completion of the fence command occurs once the
    state of the corresponding sync object has been changed, but
    commands waiting on that sync object may not be unblocked until
    some time after the fence command completes.

    <flags> must be 0[fn1].
       [fn1: <flags> is a placeholder for anticipated future extensions
    of fence sync object capabilities.]

    Each sync object contains a number of <properties> which determine
    the state of the object and the behavior of any commands associated
    with it. Each property has a <property name> and <property value>.
    The initial property values for a sync object created by FenceSyncAPPLE
    are shown in table 5.props:

    Property Name         Property Value
    --------------------  ----------------
    OBJECT_TYPE_APPLE     SYNC_FENCE_APPLE
    SYNC_CONDITION_APPLE  <condition>
    SYNC_STATUS_APPLE     UNSIGNALED_APPLE
    SYNC_FLAGS_APPLE      <flags>
    --------------------------------------
    Table 5.props: Initial properties of a
    sync object created with FenceSyncAPPLE.

    Properties of a sync object may be queried with GetSyncivAPPLE (see
    section 6.1.6). The SYNC_STATUS_APPLE property will be changed to
    SIGNALED_APPLE when <condition> is satisfied.

    If FenceSyncAPPLE fails to create a sync object, zero will be returned
    and a GL error will be generated as described. An INVALID_ENUM error
    is generated if <condition> is not SYNC_GPU_COMMANDS_COMPLETE_APPLE. If
    <flags> is not zero, an INVALID_VALUE error is generated.

    A sync object can be deleted by passing its name to the command

    void DeleteSyncAPPLE(sync sync);

    If the fence command corresponding to the specified sync object has
    completed, or if no ClientWaitSyncAPPLE or WaitSyncAPPLE commands are 
    blocking on <sync>, the object is deleted immediately. Otherwise, <sync> 
    is flagged for deletion and will be deleted when it is no longer
    associated with any fence command and is no longer blocking any
    ClientWaitSyncAPPLE or WaitSyncAPPLE command. In either case, after 
    returning from DeleteSyncAPPLE the <sync> name is invalid and can no 
    longer be used to refer to the sync object.

    DeleteSyncAPPLE will silently ignore a <sync> value of zero. An
    INVALID_VALUE error is generated if <sync> is neither zero nor the
    name of a sync object.


    5.2.1 Waiting for Sync Objects
    ------------------------------

    The command

    enum ClientWaitSyncAPPLE(sync sync, bitfield flags, uint64 timeout);

    causes the GL to block, and will not return until the sync object
    <sync> is signaled, or until the specified <timeout> period expires.
    <timeout> is in units of nanoseconds. <timeout> is adjusted to the
    closest value allowed by the implementation-dependent timeout
    accuracy, which may be substantially longer than one nanosecond, and
    may be longer than the requested period.

    If <sync> is signaled at the time ClientWaitSyncAPPLE is called
    then ClientWaitSyncAPPLE returns immediately. If <sync> is unsignaled 
    at the time ClientWaitSyncAPPLE is called then ClientWaitSyncAPPLE will 
    block and will wait up to <timeout> nanoseconds for <sync> to become 
    signaled. <flags> controls command flushing behavior, and may be
    SYNC_FLUSH_COMMANDS_BIT_APPLE, as discussed in section 5.2.2.

    ClientWaitSyncAPPLE returns one of four status values. A return value of
    ALREADY_SIGNALED_APPLE indicates that <sync> was signaled at the time
    ClientWaitSyncAPPLE was called. ALREADY_SIGNALED_APPLE will always be 
    returned if <sync> was signaled, even if the value of <timeout> is zero. 
    A return value of TIMEOUT_EXPIRED_APPLE indicates that the specified 
    timeout period expired before <sync> was signaled. A return value of
    CONDITION_SATISFIED_APPLE indicates that <sync> was signaled before the
    timeout expired. Finally, if an error occurs, in addition to
    generating a GL error as specified below, ClientWaitSyncAPPLE immediately
    returns WAIT_FAILED_APPLE without blocking.

    If the value of <timeout> is zero, then ClientWaitSyncAPPLE does not
    block, but simply tests the current state of <sync>. 
    TIMEOUT_EXPIRED_APPLE will be returned in this case if <sync> is not 
    signaled, even though no actual wait was performed.

    If <sync> is not the name of a sync object, an INVALID_VALUE error
    is generated. If <flags> contains any bits other than
    SYNC_FLUSH_COMMANDS_BIT_APPLE, an INVALID_VALUE error is generated.

    The command

    void WaitSyncAPPLE(sync sync, bitfield flags, uint64 timeout);

    is similar to ClientWaitSyncAPPLE, but instead of blocking and not
    returning to the application until <sync> is signaled, WaitSyncAPPLE
    returns immediately, instead causing the GL server [fn2] to block
    until <sync> is signaled [fn3].
       [fn2 - the GL server may choose to wait either in the CPU
    executing server-side code, or in the GPU hardware if it
    supports this operation.]
       [fn3 - WaitSyncAPPLE allows applications to continue to queue commands
    from the client in anticipation of the sync being signalled,
    increasing client-server parallelism.]

    <sync> has the same meaning as for ClientWaitSyncAPPLE.

    <timeout> must currently be the special value TIMEOUT_IGNORED_APPLE, and
    is not used. Instead, WaitSyncAPPLE will always wait no longer than an
    implementation-dependent timeout. The duration of this timeout in
    nanoseconds may be queried by calling GetInteger64vAPPLE with <value>
    MAX_SERVER_WAIT_TIMEOUT_APPLE. There is currently no way to determine
    whether WaitSyncAPPLE unblocked because the timeout expired or because
    the sync object being waited on was signaled.

    <flags> must be 0.

    If an error occurs, WaitSyncAPPLE generates a GL error as specified
    below, and does not cause the GL server to block.

    If <sync> is not the name of a sync object, an INVALID_VALUE error
    is generated. If <timeout> is not TIMEOUT_IGNORED_APPLE, or <flags> 
    is not zero, an INVALID_VALUE error is generated [fn4].
       [fn4 - <flags> and <timeout> are placeholders for anticipated future 
    extensions of sync object capabilities. They must have these reserved 
    values in order that existing code calling WaitSyncAPPLE operate properly 
    in the presence of such extensions.]

    Multiple Waiters
    ----------------

    It is possible for both the GL client to be blocked on a sync object
    in a ClientWaitSyncAPPLE command, the GL server to be blocked as the
    result of a previous WaitSyncAPPLE command, and for additional 
    WaitSyncAPPLE commands to be queued in the GL server, all for a single 
    sync object. When such a sync object is signaled in this situation, the
    client will be unblocked, the server will be unblocked, and all such
    queued WaitSyncAPPLE commands will continue immediately when they are
    reached.

    See appendix C.2 for more information about blocking on a sync
    object in multiple GL contexts.

    5.2.2 Signalling
    ----------------

    A fence sync object can be in the signaled state only once the
    corresponding fence command has completed and signaled the sync
    object.

    If the sync object being blocked upon will not be signaled in finite
    time (for example, by an associated fence command issued previously,
    but not yet flushed to the graphics pipeline), then ClientWaitSyncAPPLE
    may hang forever. To help prevent this behavior [fn5], if the
    SYNC_FLUSH_COMMANDS_BIT_APPLE bit is set in <flags>, and <sync> is
    unsignaled when ClientWaitSyncAPPLE is called, then the equivalent of
    Flush will be performed before blocking on <sync>.
       [fn5 - The simple flushing behavior defined by
    SYNC_FLUSH_COMMANDS_BIT_APPLE will not help when waiting for a fence
    command issued in another context's command stream to complete.
    Applications which block on a fence sync object must take
    additional steps to assure that the context from which the
    corresponding fence command was issued has flushed that command
    to the graphics pipeline.]

    If a sync object is marked for deletion while a client is blocking
    on that object in a ClientWaitSyncAPPLE command, or a GL server is
    blocking on that object as a result of a prior WaitSyncAPPLE command,
    deletion is deferred until the sync object is signaled and all
    blocked GL clients and servers are unblocked.

    Additional constraints on the use of sync objects are discussed in
    appendix C.

    State must be maintained to indicate which sync object names are
    currently in use. The state require for each sync object in use is
    an integer for the specific type, an integer for the condition, an
    integer for the flags, and a bit indicating whether the object is
    signaled or unsignaled. The initial values of sync object state are
    defined as specified by FenceSyncAPPLE."

    Update the Debug Labels section's last sentence to include 
    SYNC_OBJECT_APPLE as a value supported for the <type> passed to
    LabelObjectEXT.

Additions to Chapter 6 of the OpenGL ES 2.0 Specification (State and State 
Requests)

    Add GetInteger64vAPPLE to the first list of commands in section 6.1.1
    "Simple Queries", and change the next sentence to mention the query:

   "There are four commands for obtaining simple state variables:

       ...
       void GetInteger64vAPPLE(enum value, int64 *data);
       ...

    The commands obtain boolean, integer, 64-bit integer, or
    floating-point..."

    Modify the third sentence of section 6.1.2 "Data Conversions":

   "If any of the other simple queries are called, a boolean value of
    TRUE or FALSE is interpreted as 1 or 0, respectively. If GetIntegerv
    or GetInteger64vAPPLE are called, a floating-point value is rounded to
    the nearest integer, unless the value is an RGBA color component..."

    Insert a new subsection following "String Queries" (subsection
    6.1.5) describing sync object queries. Renumber existing subsection
    6.1.6 "Buffer Object Queries" and all following 6.1.* subsections.

   "6.1.6 Sync Object Queries

    Properties of sync objects may be queried using the command

    void GetSyncivAPPLE(sync sync, enum pname, sizei bufSize, 
        sizei *length, int *values);

    The value or values being queried are returned in the parameters
    <length> and <values>.

    On success, GetSyncivAPPLE replaces up to <bufSize> integers in <values>
    with the corresponding property values of the object being queried.
    The actual number of integers replaced is returned in *<length>. If
    <length> is NULL, no length is returned.

    If <pname> is OBJECT_TYPE_APPLE, a single value representing the specific
    type of the sync object is placed in <values>. The only type
    supported is SYNC_FENCE_APPLE.

    If <pname> is SYNC_STATUS_APPLE, a single value representing the status of
    the sync object (SIGNALED_APPLE or UNSIGNALED_APPLE) is placed in <values>.

    If <pname> is SYNC_CONDITION_APPLE, a single value representing the
    condition of the sync object is placed in <values>. The only
    condition supported is SYNC_GPU_COMMANDS_COMPLETE_APPLE.

    If <pname> is SYNC_FLAGS_APPLE, a single value representing the flags with
    which the sync object was created is placed in <values>. No flags
    are currently supported.

    If <sync> is not the name of a sync object, an INVALID_VALUE error
    is generated. If <pname> is not one of the values described above,
    an INVALID_ENUM error is generated. If an error occurs,
    nothing will be written to <values> or <length>.

    The command

    boolean IsSyncAPPLE(sync sync);

    returns TRUE if <sync> is the name of a sync object. If <sync> is
    not the name of a sync object, or if an error condition occurs,
    IsSyncAPPLE returns FALSE (note that zero is not the name of a sync
    object).

    Sync object names immediately become invalid after calling
    DeleteSyncAPPLE, as discussed in sections 5.2 and D.2, but the underlying
    sync object will not be deleted until it is no longer associated
    with any fence command and no longer blocking any *WaitSyncAPPLE command."

    Update the Debug Labels section's last sentence to include 
    SYNC_OBJECT_APPLE as a value supported for the <type> passed to
    GetObjectLabelEXT.


Additions to Appendix C (Shared Objects and Multiple Contexts)

    In the third paragraph of the appendix, add "sync objects" to the
    list of shared state.

    Add "sync objects" to the list of objects in the first sentence of
    C.1.3.  Append the following to the first paragraph of C.1.3:
    
   "A sync object is in use while there is a corresponding fence command 
    which has not yet completed and signaled the sync object, or while 
    there are any GL clients and/or servers blocked on the sync object as 
    a result of ClientWaitSyncAPPLE or WaitSyncAPPLE commands."
    
   "C.1.3 Deleted Object and Object Name Lifetimes

    Insert a new section following "Object Deletion Behavior" (section
    C.1) describing sync object multicontext behavior. Renumber existing
    section C.2 "Propagating State Changes..." and all following C.*
    sections.

   "C.2 Sync Objects and Multiple Contexts
    --------------------------------------
    
    When multiple GL clients and/or servers are blocked on a single sync 
    object and that sync object is signalled, all such blocks are released. 
    The order in which blocks are released is implementation-dependent."

    Edit the former C.2.1 to read as follows:

   "C.3.1 Determining Completion of Changes to an object
    ----------------------------------------------------

    The contents of an object T are considered to have been changed once 
    a command such as described in section C.2 has completed. Completion 
    of a command [fn6] may be determined either by calling Finish, or by 
    calling FenceSyncAPPLE and executing a WaitSyncAPPLE command on the 
    associated sync object. The second method does not require a round trip 
    to the GL server and may be more efficient, particularly when changes to 
    T in one context must be known to have completed before executing 
    commands dependent on those changes in another context.
        [fn6: The GL already specifies that a single context processes
    commands in the order they are received. This means that a change to 
    an object in a context at time <t> must be completed by the time a 
    command issued in the same context at time <t+1> uses the result of that 
    change.]"

Dependencies on EXT_debug_label

    If EXT_debug_label is not available, omit the updates to Debug label
    sections of chapters 5 and 6, and omit SYNC_OBJECT_APPLE from Table 6.X.

Errors

    INVALID_VALUE is generated if the <sync> parameter of
    ClientWaitSyncAPPLE, WaitSyncAPPLE, or GetSyncivAPPLE is not the name
    of a sync object.

    INVALID_VALUE is generated if the <sync> parameter of DeleteSyncAPPLE
    is neither zero nor the name of a sync object.

    INVALID_ENUM is generated if the <condition> parameter of FenceSyncAPPLE
    is not SYNC_GPU_COMMANDS_COMPLETE_APPLE.

    INVALID_VALUE is generated if the <flags> parameter of
    ClientWaitSyncAPPLE contains bits other than SYNC_FLUSH_COMMANDS_BIT_APPLE, 
    or if the <flags> parameter of WaitSyncAPPLE is nonzero.

    INVALID_ENUM is generated if the <pname> parameter of GetSyncivAPPLE is
    neither OBJECT_TYPE_APPLE, SYNC_CONDITION_APPLE, SYNC_FLAGS_APPLE, nor 
    SYNC_STATUS_APPLE.

New State

Table 6.X.  Sync Objects.

Get value             Type  Get command        Initial value                     Description            Section
--------------------  ----  -----------------  --------------------------------  ---------------------  -------
OBJECT_TYPE_APPLE     Z_1   GetSyncivAPPLE     SYNC_FENCE_APPLE                  Type of sync object    5.2
SYNC_STATUS_APPLE     Z_2   GetSyncivAPPLE     UNSIGNALED_APPLE                  Sync object status     5.2
SYNC_CONDITION_APPLE  Z_1   GetSyncivAPPLE     SYNC_GPU_COMMANDS_COMPLETE_APPLE  Sync object condition  5.2
SYNC_FLAGS_APPLE      Z     GetSyncivAPPLE     SYNC_FLAGS_APPLE                  Sync object flags      5.2
SYNC_OBJECT_APPLE     0*xc  GetObjectLabelEXT  empty                             Debug label            5.X

New Implementation Dependent State

Table 40. Implementation Dependent Values (cont.)

Get value           Type  Get command    Min. value  Description       Section
------------------  ----  -------------  ----------  ----------------  -------
MAX_SERVER_WAIT_    Z^+   GetInteger64v  0           Maximum WaitSync  5.2
TIMEOUT_APPLE                                         timeout interval

Sample Code

    ... kick off a lengthy GL operation
    /* Place a fence and associate a fence sync object */
    GLsyncAPPLE sync = glFenceSyncAPPLE(GLSYNC_GPU_COMMANDS_COMPLETE_APPLE, 0);

    /* Elsewhere, wait for the sync to be signaled */

    /* To wait a specified amount of time (possibly clamped), do
     * this to convert a time in seconds into nanoseconds:
     */
    GLuint64APPLE timeout = num_seconds * ((GLuint64)1000 * 1000 * 1000);
    glWaitSyncAPPLE(sync, 0, timeout);

    /* Or to determine the maximum possible wait interval and wait
     * that long, do this instead:
     */
    GLuint64APPLE max_timeout;
    glGetInteger64vAPPLE(GL_MAX_SERVER_WAIT_TIMEOUT_APPLE, &max_timeout);
    glWaitSyncAPPLE(sync, 0, max_timeout);

Issues

    See ARB_sync issues list: http://www.opengl.org/registry/specs/ARB/sync.txt

Revision History

    Version 3, 2012/07/10 - Add support for debug label extension.
    Version 2, 2012/06/18 - Correct spec to indicate ES 1.1 may also be okay.
    Version 1, 2012/06/01 - Conversion from ARB_sync to APPLE_sync for ES.
