# EXT_disjoint_timer_query

Name

    EXT_disjoint_timer_query

Name Strings

    GL_EXT_disjoint_timer_query

Contact

    Maurice Ribble, Qualcomm (mribble 'at' qualcomm.com)

Contributors

    Matt Trusten
    Maurice Ribble
    Daniel Koch
    Jan-Harald Fredriksen

    Contributors to ANGLE_timer_query
    Contributors to ARB_timer_query
    Contributors to EXT_timer_query
    Contributors to EXT_occlusion_query_boolean
    This extension is based on the ARB_timer_query and ANGLE_timer_query

IP Status

    No known IP claims.

Status

    Complete

Version

    Version 9, November 20, 2020

Number

    OpenGL ES Extension #150

Dependencies

    OpenGL ES 2.0 or greater required.

    The extension is written against the OpenGL ES 2.0 specification.
    
    The GetInteger64vEXT command is required only if OpenGL ES 3.0 or
    later is not supported (see the Interactions section for details).

Overview

    Applications can benefit from accurate timing information in a number of
    different ways.  During application development, timing information can
    help identify application or driver bottlenecks.  At run time,
    applications can use timing information to dynamically adjust the amount
    of detail in a scene to achieve constant frame rates.  OpenGL
    implementations have historically provided little to no useful timing
    information.  Applications can get some idea of timing by reading timers
    on the CPU, but these timers are not synchronized with the graphics
    rendering pipeline.  Reading a CPU timer does not guarantee the completion
    of a potentially large amount of graphics work accumulated before the
    timer is read, and will thus produce wildly inaccurate results.
    glFinish() can be used to determine when previous rendering commands have
    been completed, but will idle the graphics pipeline and adversely affect
    application performance.

    This extension provides a query mechanism that can be used to determine
    the amount of time it takes to fully complete a set of GL commands, and
    without stalling the rendering pipeline.  It uses the query object
    mechanisms first introduced in the occlusion query extension, which allow
    time intervals to be polled asynchronously by the application.

New Procedures and Functions

    void GenQueriesEXT(sizei n, uint *ids);
    void DeleteQueriesEXT(sizei n, const uint *ids);
    boolean IsQueryEXT(uint id);
    void BeginQueryEXT(enum target, uint id);
    void EndQueryEXT(enum target);
    void QueryCounterEXT(uint id, enum target);
    void GetQueryivEXT(enum target, enum pname, int *params);
    void GetQueryObjectivEXT(uint id, enum pname, int *params);
    void GetQueryObjectuivEXT(uint id, enum pname, uint *params);
    void GetQueryObjecti64vEXT(uint id, enum pname, int64 *params);
    void GetQueryObjectui64vEXT(uint id, enum pname, uint64 *params);
    void GetInteger64vEXT(enum pname, int64 *data)

New Tokens

    Accepted by the <pname> parameter of GetQueryivEXT:

        QUERY_COUNTER_BITS_EXT                       0x8864
        CURRENT_QUERY_EXT                            0x8865

    Accepted by the <pname> parameter of GetQueryObjectivEXT,
    GetQueryObjectuivEXT, GetQueryObjecti64vEXT, and
    GetQueryObjectui64vEXT:

        QUERY_RESULT_EXT                             0x8866
        QUERY_RESULT_AVAILABLE_EXT                   0x8867
        
    Accepted by the <target> parameter of BeginQueryEXT, EndQueryEXT, and
    GetQueryivEXT:

        TIME_ELAPSED_EXT                             0x88BF

    Accepted by the <target> parameter of GetQueryiv and QueryCounter.
    Accepted by the <value> parameter of GetBooleanv, GetIntegerv,
    GetInteger64v, and GetFloatv:

        TIMESTAMP_EXT                                0x8E28

    Accepted by the <value> parameter of GetBooleanv, GetIntegerv,
    GetInteger64v, and GetFloatv:

        GPU_DISJOINT_EXT                             0x8FBB

Additions to Chapter 2 of the OpenGL ES 2.0 Specification (OpenGL ES Operation)

    (Modify table 2.1, Correspondence of command suffix letters to GL argument)
    Add two new types:
    
    Letter Corresponding GL Type
    ------ ---------------------
    i64    int64
    ui64   uint64

    (Modify table 2.2, GL data types) Add two new types:
    
    GL Type    Minimum Bit Width   Description
    -------    -----------------   -----------------------------
    int64      64                  Signed 2's complement integer
    uint64     64                  Unsigned binary integer

Additions to Chapter 5 of the OpenGL ES 2.0 Specification (Special Functions)

    Add a new section 5.3 "Timer Queries":

    "5.3  Timer Queries

    Timer queries use query objects to track the amount of time needed to
    fully complete a set of GL commands, or to determine the current time
    of the GL.
    
    Timer queries are associated with query objects.  The command

      void GenQueriesEXT(sizei n, uint *ids);

    returns <n> previously unused query object names in <ids>.  These
    names are marked as used, but no object is associated with them until
    the first time they are used by BeginQueryEXT or QueryCounterEXT.  Query 
    objects contain one piece of state, an integer result value.  This result 
    value is initialized to zero when the object is created.  Any positive 
    integer except for zero (which is reserved for the GL) is a valid query
    object name.

    Query objects are deleted by calling

      void DeleteQueriesEXT(sizei n, const uint *ids);

    <ids> contains <n> names of query objects to be deleted.  After a
    query object is deleted, its name is again unused.  Unused names in
    <ids> are silently ignored.
    If an active query object is deleted its name immediately becomes unused,
    but the underlying object is not deleted until it is no longer active.

    A timer query can be started and finished by calling

      void BeginQueryEXT(enum target, uint id);
      void EndQueryEXT(enum target);

    where <target> is TIME_ELAPSED_EXT.  If BeginQueryEXT is called
    with an unused <id>, that name is marked as used and associated with
    a new query object.
    
    If BeginQueryEXT is called with an <id> of zero, if the active query
    object name for <target> is non-zero, if <id> is the name of an existing
    query object whose type does not match <target>, or if <id> is the active
    query object name for any query type, the error INVALID_OPERATION is
    generated.  If EndQueryEXT is called while no query with the same target
    is in progress, an INVALID_OPERATION error is generated.

    When BeginQueryEXT and EndQueryEXT are called with a <target> of
    TIME_ELAPSED_EXT, the GL prepares to start and stop the timer used for
    timer queries.  The timer is started or stopped when the effects from all
    previous commands on the GL client and server state and the framebuffer
    have been fully realized.  On some architectures framebuffer can have 
    different meanings (specifically on some tiling GPUs fully realized might refer
    to the framebuffer being in tile memory before it gets copied out to system 
    memory).  The BeginQueryEXT and EndQueryEXT commands may return before the
    timer is actually started or stopped.  When the timer query timer is finally
    stopped, the elapsed time (in nanoseconds) is written to the corresponding
    query object as the query result value, and the query result for that object
    is marked as available.

    If the elapsed time overflows the number of bits, <n>, available to hold
    elapsed time, its value becomes undefined.  It is recommended, but not
    required, that implementations handle this overflow case by saturating at
    2^n - 1.

    The necessary state is a single bit indicating whether a timer
    query is active, the identifier of the currently active timer
    query, and a counter keeping track of the time that has passed.

    When the command

         void QueryCounterEXT(uint id, enum target);

    is called with <target> TIMESTAMP_EXT, the GL records the current time
    into the corresponding query object. The time is recorded after all
    previous commands on the GL client and server state and the framebuffer
    have been fully realized.  On some architectures framebuffer can have 
    different meanings (specifically on some tiling GPUs fully realized might refer 
    to the framebuffer being in tile memory before it gets copied out to system 
    memory).  When the time is recorded, the query result for that object is 
    marked available. QueryCounterEXT timer queries can be used within a 
    BeginQueryEXT / EndQueryEXT block where the <target> is TIME_ELAPSED_EXT and
    it does not affect the result of that query object.  The error 
    INVALID_OPERATION is generated if the <id> is already in use within a 
    BeginQueryEXT/EndQueryEXT block.

    The current time of the GL may be queried by calling GetIntegerv or
    GetInteger64v with the symbolic constant TIMESTAMP_EXT. This will return
    the GL time after all previous commands have reached the GL server but have
    not yet necessarily executed. By using a combination of this synchronous
    get command and the asynchronous timestamp query object target,
    applications can measure the latency between when commands reach the GL
    server and when they are realized in the framebuffer.
    
    In order to know if the value returned from GetIntegerv or GetQuery is valid
    GPU_DISJOINT_EXT needs to be used to make sure the GPU did not perform any
    disjoint operation. This can be done through GetIntegerv by using GPU_-
    DISJOINT_EXT for <pname>. <params> will be filled with a non-zero value if
    a disjoint operation occurred since the last time GetIntegerv was used with
    GPU_DISJOINT_EXT. A zero value will be returned if no disjoint operation
    occurred, indicating the values returned by this extension that are found
    in-between subsequent GetIntegerv calls will be valid for performance
    metrics.

    Disjoint operations occur whenever a change in the GPU occurs that will
    make the values returned by this extension unusable for performance 
    metrics. An example can be seen with how mobile GPUs need to proactively
    try to conserve power, which might cause the GPU to go to sleep at the 
    lower levers. This means disjoint states will occur at different times on
    different platforms and are implementation dependent. When the returned
    value is non-zero, all time values that were filled since the previous 
    disjoint check should be considered undefined."

Additions to Chapter 6 of the OpenGL ES 2.0 Specification (State and State
Requests)

    Add GetInteger64vEXT to section 6.1.1 "Simple Queries", following
    the prototype for GetIntegerv:

    "  void GetInteger64vEXT(enum pname, int64 *data);
       void GetFloatv(enum value, float *data);

    The commands obtain boolean, integer, 64-bit integer, or floating-point
    variables..."

    Add a new section 6.1.9 "Timer Queries":

    "The command

      boolean IsQueryEXT(uint id);

    returns TRUE if <id> is the name of a query object.  If <id> is zero,
    or if <id> is a non-zero value that is not the name of a query
    object, IsQueryEXT returns FALSE.

    Information about a query target can be queried with the command

      void GetQueryivEXT(enum target, enum pname, int *params);

    <target> identifies the query target and can be TIME_ELAPSED_EXT or
    TIMESTAMP_EXT for timer queries.

    If <pname> is CURRENT_QUERY_EXT, the name of the currently active query
    for <target>, or zero if no query is active, will be placed in <params>.

    If <pname> is QUERY_COUNTER_BITS_EXT, the implementation-dependent number
    of bits used to hold the query result for <target> will be placed in
    <params>.  The number of query counter bits may be zero, in which case
    the counter contains no useful information.

    For timer queries (TIME_ELAPSED_EXT and TIMESTAMP_EXT), if the number
    of bits is non-zero, the minimum number of bits allowed is 30 which
    will allow at least 1 second of timing.

    The state of a query object can be queried with the commands

      void GetQueryObjectivEXT(uint id, enum pname, int *params);
      void GetQueryObjectuivEXT(uint id, enum pname, uint *params);
      void GetQueryObjecti64vEXT(uint id, enum pname, int64 *params);
      void GetQueryObjectui64vEXT(uint id, enum pname, uint64 *params);

    If <id> is not the name of a query object, or if the query object
    named by <id> is currently active, then an INVALID_OPERATION error is
    generated.

    If <pname> is QUERY_RESULT_EXT, then the query object's result
    value is returned as a single integer in <params>. If the value is so
    large in magnitude that it cannot be represented with the requested type,
    then the nearest value representable using the requested type is
    returned. If the number of query counter bits for target is zero, then
    the result is returned as a single integer with the value zero.
    
    There may be an indeterminate delay before the above query returns. If
    <pname> is QUERY_RESULT_AVAILABLE_EXT, FALSE is returned if such a delay
    would be required; otherwise TRUE is returned. It must always be true
    that if any query object returns a result available of TRUE, all queries
    of the same type issued prior to that query must also return TRUE.

    Querying the state for a given timer query forces that timer query to
    complete within a finite amount of time.

    If multiple queries are issued on the same target and id prior to 
    calling GetQueryObject[u]i[64]vEXT, the result returned will always be
    from the last query issued.  The results from any queries before the
    last one will be lost if the results are not retrieved before starting
    a new query on the same <target> and <id>.

Errors

    The error INVALID_VALUE is generated if GenQueriesEXT is called where
    <n> is negative.

    The error INVALID_VALUE is generated if DeleteQueriesEXT is called
    where <n> is negative.

    The error INVALID_OPERATION is generated if BeginQueryEXT is called
    when a query of the given <target> is already active.

    The error INVALID_OPERATION is generated if EndQueryEXT is called
    when a query of the given <target> is not active.

    The error INVALID_OPERATION is generated if BeginQueryEXT is called
    where <id> is zero.

    The error INVALID_OPERATION is generated if BeginQueryEXT is called
    where <id> is the name of a query currently in progress.
    
    The error INVALID_OPERATION is generated if BeginQueryEXT is called
    where <id> is the name of an existing query object whose type does not
    match <target>.

    The error INVALID_ENUM is generated if BeginQueryEXT or EndQueryEXT
    is called where <target> is not TIME_ELAPSED_EXT.

    The error INVALID_ENUM is generated if GetQueryivEXT is called where
    <target> is not TIME_ELAPSED_EXT or TIMESTAMP_EXT.

    The error INVALID_ENUM is generated if GetQueryivEXT is called where
    <pname> is not QUERY_COUNTER_BITS_EXT or CURRENT_QUERY_EXT.

    The error INVALID_ENUM is generated if QueryCounterEXT is called where
    <target> is not TIMESTAMP_EXT.

    The error INVALID_OPERATION is generated if QueryCounterEXT is called
    on a query object that is already in use inside a
    BeginQueryEXT/EndQueryEXT.

    The error INVALID_OPERATION is generated if GetQueryObjectivEXT,
    GetQueryObjectuivEXT, GetQueryObjecti64vEXT, or
    GetQueryObjectui64vEXT is called where <id> is not the name of a query
    object.

    The error INVALID_OPERATION is generated if GetQueryObjectivEXT,
    GetQueryObjectuivEXT, GetQueryObjecti64vEXT, or
    GetQueryObjectui64vEXT is called where <id> is the name of a currently
    active query object.

    The error INVALID_ENUM is generated if GetQueryObjectivEXT,
    GetQueryObjectuivEXT, GetQueryObjecti64vEXT, or
    GetQueryObjectui64vEXT is called where <pname> is not
    QUERY_RESULT_EXT or QUERY_RESULT_AVAILABLE_EXT.

New State

    (Add a new table 6.xx, "Query Operations")
    
    Get Value                      Type    Get Command              Initial Value   Description              Sec
    ---------                      ----    -----------              -------------   -----------              ------
    -                              B       -                        FALSE           query active             5.3
    CURRENT_QUERY_EXT              Z+      GetQueryivEXT            0               active query ID          5.3
    QUERY_RESULT_EXT               Z+      GetQueryObjectuivEXT,    0               samples-passed count     5.3
                                           GetQueryObjectui64vEXT
    QUERY_RESULT_AVAILABLE_EXT     B       GetQueryObjectivEXT      FALSE           query result available   5.3

New Implementation Dependent State

    (Add the following entry to table 6.18):

    Get Value                      Type    Get Command      Minimum Value      Description           Sec
    --------------------------     ----    -----------      -------------      ----------------      ------
    QUERY_COUNTER_BITS_EXT         Z+      GetQueryivEXT    see 6.1.9          Number of bits in     6.1.9
                                                                               query counter

Interactions with OpenGL ES 2.0 and OpenGL ES 3.x

    If only OpenGL ES 2.0 is supported, then GetInteger64vEXT is defined,
    and is used instead of the GetInteger64v command defined by OpenGL ES
    3.x. If OpenGL ES 3.0 or later is supported, GetInteger64vEXT is not
    required by an implementation of this extension, and the changes to
    section 6.1.1 are ignored.

Examples

    (1) Here is some rough sample code that demonstrates the intended usage
        of this extension.

        GLint queries[N];
        GLint available = 0;
        GLint disjointOccurred = 0;
        /* Timer queries can contain more than 32 bits of data, so always
           query them using the 64 bit types to avoid overflow */
        GLuint64 timeElapsed = 0;

        /* Create a query object. */
        glGenQueries(N, queries);
        
        /* Clear disjoint error */
        glGetIntegerv(GL_GPU_DISJOINT_EXT, &disjointOccurred);

        /* Start query 1 */
        glBeginQuery(GL_TIME_ELAPSED_EXT, queries[0]);

        /* Draw object 1 */
        ....

        /* End query 1 */
        glEndQuery(GL_TIME_ELAPSED_EXT);

        ...

        /* Start query N */
        glBeginQuery(GL_TIME_ELAPSED_EXT, queries[N-1]);

        /* Draw object N */
        ....

        /* End query N */
        glEndQuery(GL_TIME_ELAPSED_EXT);

        /* Wait for all results to become available */
        while (!available) {
            glGetQueryObjectiv(queries[N-1], GL_QUERY_RESULT_AVAILABLE, &available);
        }
        
        /* Check for disjoint operation for all queries within the last
           disjoint check. This way we can only check disjoint once for all
           queries between, and once the last is filled we know all previous
           will have been filled as well */
        glGetIntegerv(GL_GPU_DISJOINT_EXT, &disjointOccurred);
        
        /* If a disjoint operation occurred, all timer queries in between
           the last two disjoint checks that were filled are invalid, continue
           without reading the the values */
        if (!disjointOccurred) {
            for (i = 0; i < N; i++) {
                /* See how much time the rendering of object i took in nanoseconds. */
                glGetQueryObjectui64vEXT(queries[i], GL_QUERY_RESULT, &timeElapsed);
                
                /* Do something useful with the time if a disjoint operation did
                   not occur.  Note that care should be taken to use all
                   significant bits of the result, not just the least significant
                   32 bits. */
                AdjustObjectLODBasedOnDrawTime(i, timeElapsed);
            }
        }

        This example is sub-optimal in that it stalls at the end of every
        frame to wait for query results.  Ideally, the collection of results
        would be delayed one frame to minimize the amount of time spent
        waiting for the GPU to finish rendering.
        
    (2) This example uses QueryCounter.
    
        GLint queries[2];
        GLint available = 0;
        GLint disjointOccurred = 0;
        /* Timer queries can contain more than 32 bits of data, so always
           query them using the 64 bit types to avoid overflow */
        GLuint64 timeStart, timeEnd, timeElapsed = 0;

        /* Create a query object. */
        glGenQueries(2, queries);

        /* Clear disjoint error */
        glGetIntegerv(GL_GPU_DISJOINT_EXT, &disjointOccurred);

        /* Query current timestamp before drawing */
        glQueryCounterEXT(queries[0], GL_TIMESTAMP_EXT);

        /* Draw full rendertarget of objects */

        /* Query current timestamp after drawing */
        glQueryCounterEXT(queries[1], GL_TIMESTAMP_EXT);
        
        /* Do some other work so you don't stall waiting for available */
        
        /* Wait for the query result to become available */
        while (!available) {
            glGetQueryObjectiv(queries[1], GL_QUERY_RESULT_AVAILABLE, &available);
        }
        
        /* Check for disjoint operation. */
        glGetIntegerv(GL_GPU_DISJOINT_EXT, &disjointOccurred);
        
        /* If a disjoint operation occurred, continue without reading the the
           values */
        if (!disjointOccurred) {
            /* Get timestamp for when rendertarget started. */
            glGetQueryObjectui64vEXT(queries[0], GL_QUERY_RESULT, &timeStart);
            /* Get timestamp for when rendertarget finished. */
            glGetQueryObjectui64vEXT(queries[1], GL_QUERY_RESULT, &timeEnd);
            /* See how much time the rendering took in nanoseconds. */
            timeElapsed = timeEnd - timeStart;
            
            /* Do something useful with the time if a disjoint operation did
               not occur.  Note that care should be taken to use all
               significant bits of the result, not just the least significant
               32 bits. */
            AdjustObjectLODBasedOnDrawTime(timeElapsed);
        }
        
    (3) This example demonstrates how to measure the latency between GL
        commands reaching the server and being realized in the framebuffer.
        
        /* Submit a frame of rendering commands */
        while (!doneRendering) {
            ...
            glDrawElements(...);
        }

        /* Measure rendering latency:
           Some commands may have already been submitted to hardware,
           and some of those may have already completed.  The goal is
           to measure the time it takes for the remaining commands to
           complete, thereby measuring how far behind the app the GPU
           is lagging, but without synchronizing the GPU with the CPU. */

        /* Clear disjoint error */
        glGetIntegerv(GL_GPU_DISJOINT_EXT, &disjointOccurred);
        
        /* Queue a query to find out when the frame finishes on the GL */
        glQueryCounterEXT(endFrameQuery, GL_TIMESTAMP_EXT);

        /* Get the current GL time without stalling the GL */
        glGetIntegerv(GL_TIMESTAMP_EXT, &flushTime);

        /* Finish the frame, submitting outstanding commands to the GL */
        SwapBuffers();

        /* Render another frame */

        /* Later, compare the query result of <endFrameQuery>
           and <flushTime> to measure the latency of the frame.
           A disjoint error still needs to be checked for in order 
           to verify these values are valid. */

Issues from EXT_timer_query

    (1) What time interval is being measured?

    RESOLVED:  The timer starts when all commands prior to BeginQueryEXT() have
    been fully executed.  At that point, everything that should be drawn by
    those commands has been written to the framebuffer.  The timer stops
    when all commands prior to EndQueryEXT() have been fully executed.

    (2) What unit of time will time intervals be returned in?

    RESOLVED:  Nanoseconds (10^-9 seconds).  This unit of measurement allows
    for reasonably accurate timing of even small blocks of rendering
    commands.  The granularity of the timer is implementation-dependent.  A
    32-bit query counter can express intervals of up to approximately 4
    seconds.

    (3) What should be the minimum number of counter bits for timer queries?

    RESOLVED:  30 bits, which will allow timing sections that take up to 1
    second to render.

    (4) How are counter results of more than 32 bits returned?

    RESOLVED:  Via two new datatypes, int64 and uint64, and their
    corresponding GetQueryObject entry points.  These types hold integer
    values and have a minimum bit width of 64.

    (5) Should the extension measure total time elapsed between the full
        completion of the BeginQuery and EndQuery commands, or just time
        spent in the graphics library?

    RESOLVED:  This extension will measure the total time elapsed between
    the full completion of these commands.  Future extensions may implement
    a query to determine time elapsed at different stages of the graphics
    pipeline.

    (6) If multiple query types are supported, can multiple query types be
        active simultaneously?

    RESOLVED:  Yes; an application may perform a timer query and another
    type of query simultaneously.  An application can not perform multiple
    timer queries or multiple queries of other types simultaneously.  An
    application also can not use the same query object for another query
    and a timer query simultaneously.

    (7) Do query objects have a query type permanently associated with them?

    RESOLVED:  No.  A single query object can be used to perform different
    types of queries, but not at the same time.

    Having a fixed type for each query object simplifies some aspects of the
    implementation -- not having to deal with queries with different result
    sizes, for example.  It would also mean that BeginQuery() with a query
    object of the "wrong" type would result in an INVALID_OPERATION error.

    UPDATE: This resolution was relevant for EXT_timer_query and OpenGL 2.0.
    Since EXT_transform_feedback has since been incorporated into the core,
    the resolution is that BeginQuery will generate error INVALID_OPERATION
    if <id> represents a query object of a different type.

    (8) How predictable/repeatable are the results returned by the timer
        query?

    RESOLVED:  In general, the amount of time needed to render the same
    primitives should be fairly constant.  But there may be many other
    system issues (e.g., context switching on the CPU and GPU, virtual
    memory page faults, memory cache behavior on the CPU and GPU) that can
    cause times to vary wildly.

    Note that modern GPUs are generally highly pipelined, and may be
    processing different primitives in different pipeline stages
    simultaneously.  In this extension, the timers start and stop when the
    BeginQuery/EndQuery commands reach the bottom of the rendering pipeline.
    What that means is that by the time the timer starts, the GL driver on
    the CPU may have started work on GL commands issued after BeginQuery,
    and the higher pipeline stages (e.g., vertex transformation) may have
    started as well.

   (9) What should the new 64 bit integer type be called?

    RESOLVED: The new types will be called GLint64/GLuint64.  The new
    command suffixes will be i64 and ui64.  These names clearly convey the
    minimum size of the types.  These types are similar to the C99 standard
    type int_least64_t, but we use names similar to the C99 optional type
    int64_t for simplicity.

Issues from ARB_timer_query

   (10) What about tile-based implementations? The effects of a command are
        not complete until the frame is completely rendered. Timing recorded
        before the frame is complete may not be what developers expect. Also
        the amount of time needed to render the same primitives is not
        consistent, which conflicts with issue (8) above. The time depends on
        how early or late in the scene it is placed.

    RESOLVED: The current language supports tile-based rendering okay as it
    is written. Developers are warned that using timers on tile-based
    implementation may not produce results they expect since rendering is not
    done in a linear order. Timing results are calculated when the frame is
    completed and may depend on how early or late in the scene it is placed.
    
   (11) Can the GL implementation use different clocks to implement the
        TIME_ELAPSED and TIMESTAMP queries?

    RESOLVED: Yes, the implementation can use different internal clocks to
    implement TIME_ELAPSED and TIMESTAMP. If different clocks are
    used it is possible there is a slight discrepancy when comparing queries
    made from TIME_ELAPSED and TIMESTAMP; they may have slight
    differences when both are used to measure the same sequence. However, this
    is unlikely to affect real applications since comparing the two queries is
    not expected to be useful.

Issues

    (12) What should we call this extension?

    RESOLVED: EXT_disjoint_timer_query

    (13) Why is this done as a separate extension instead of just supporting
         ARB_timer_query?

    ARB_timer_query is written against OpenGL 3.2, which includes a lot of
    the required support for dealing with query objects. None of these
    functions or tokens exist in OpenGL ES, and as such have to be added in
    this specification.

    (14) How does this extension differ from ARB_timer_query?

    This extension contains most ARB_timer_query behavior unchanged as well
    as adds the ability to detect GPU issues using GPU_DISJOINT_EXT.
    
    (15) Are query objects shareable between multiple contexts?

    RESOLVED: No.  Query objects are lightweight and we normally share 
    large data across contexts.  Also, being able to share query objects
    across contexts is not particularly useful.  In order to do the async 
    query across contexts, a query on one context would have to be finished 
    before the other context could query it. 
    
    (16) How does this extension interact with EXT_occlusion_query_boolean?
    
    This extension redefines the Query Api originally defined in the EXT-
    _occlusion_query_boolean. If both EXT_disjoint_timer_query and EXT-
    _occlusion_query_boolean are supported, the rules and specification 
    regarding any overlap will be governed by the EXT_occlusion_query_boolean 
    extension. 
    
    This extension should redefine the functionality in the same way, but if 
    some discrepancy is found and both are supported EXT_disjoint_timer_query 
    will yield to the rules and specifications governing the overlap in the 
    order above.

    (17) How does this extension interact with the OpenGL ES 3.0 specification?

    Some of the functionality and requirements described here overlap with the
    OpenGL ES 3.0 specification. Any overlap for the functions or tokens in 
    this extension were meant to complement each other, but the OpenGL ES 3.0 
    spec takes precedence. If the implementation supports OpenGL ES 3.0 then 
    it should support both the core non-decorated functions and the EXT
    decorated functions.
    
    (18) How do times from BeginQueryEXT/EndQueryEXT with a <target> of 
    TIME_ELAPSED_EXT and QueryCounterEXT with a <target> of TIMESTAMP_EXT 
    compare on some Qualcomm and ARM tiling GPUs?
    
    This does not describe all tiling GPUs, but it is how some tiling GPUs from
    ARM, Qualcomm, and possibly other vendors work.  This is just an
    implementation note and there is no guarantee all ARM and Qualcomm 
    implementations will work this way.
    
    TIME_ELAPSED_EXT will be a summation of all the time spent on the workload
    between begin and end.  Tiling architectures might split this work up over a
    binning pass and rendering many different tiles. It is up to the hardware 
    and/or driver to add up all the time spent on the work between begin and end
    and report a single number making the implementation transparent to
    developers using this feature.  If the binning pass happens in parallel to 
    rendering pass this time would not be counted twice.  On some
    implementations this does not include the time to copy tile memory to or from
    the frame buffer in system memory, and on other implementations this time 
    is included.
    
    TIMESTAMP_EXT is the time when all the commands are complete and copied out
    of tile memory to the framebuffer in system memory.  This can result in a 
    courser grain timestamp than developers familiar with immediate GPUs expect.
    For example all the draws to an FBO can often all get the same timestamp, or
    even all the draw calls to multiple FBOs can end up with the same timestamp.
    If some operation causes a midframe store/load (such as flush or readPixels)
    then that would create another point for timestamps, but is a lot of extra
    work for the GPU so it should be avoided.
    
    If a preemption event happens before the TIMESTAMP_EXT is reported then that
    time will include the time for preemption.  With TIME_ELAPSED_EXT it is
    undefined if the preemption time is counted or not.  Some hardware will
    count the preemption time (even though it is from a different context).  For
    this behavior GPU_DISJOINT_EXT will be set so you know there was an event
    from a different context affecting results.  Other hardware will not count 
    the time spent in the preempting context and for this cases GPU_DISJOINT_EXT
    will not be set.

Revision History
    Revision 9, 2020/11/20 (xndcn)
      - Minor fix of code sample
    Revision 8, 2019/12/11 (Jon Leech)
      - Add actual spec language defining GetInteger64vEXT (github
        OpenGL-Registry issue 326)
    Revision 7, 2016/9/2 (Maurice Ribble)
      - Clarify language dealing with GetInteger64v
    Revision 6, 2016/7/15 (Maurice Ribble)
      - Clarified some outstanding questions about tiling GPUs
      - Added issue 18
    Revision 5, 2013/6/5
      - Minor cleanup to match new gl2ext.h
    Revision 4, 2013/4/25 (Jon Leech)
      - Cleanup for publication 
      - Fix value assigned to GPU_DISJOINT_EXT
    Revision 3, 2013/4/8
      - Minor cleanup of code sample and re-wording
    Revision 2, 2013/4/2
      - Minor cleanup
    Revision 1, 2013/1/2
      - Copied from revision 1 of ANGLE_timer_query
      - Added TIMESTAMP_EXT and GPU_DISJOINT_EXT

# 用法与注意事项

## 用法与示例代码

### 获取函数指针

在Android NDK环境下，include相关egl, gles库之前，先定义宏：

`#define GL_GLEXT_PROTOTYPES 1`

即可得用相应的函数声明和宏定义。

但是链接时会出错，所以需要在运行时通过eglGetProcAddress获取函数指针，下述代码没有判断函数指针有效性，需自行判断指针为空：


```c++

decltype(glGenQueriesEXT) *glGenQueriesEXT1 = (decltype(glGenQueriesEXT)*)eglGetProcAddress("glGenQueriesEXT");
decltype(glBeginQueryEXT) *glBeginQueryEXT1 = (decltype(glBeginQueryEXT)*)eglGetProcAddress("glBeginQueryEXT");
decltype(glEndQueryEXT) *glEndQueryEXT1 = (decltype(glEndQueryEXT)*)eglGetProcAddress("glEndQueryEXT");
decltype(glDeleteQueriesEXT) *glDeleteQueriesEXT1 = (decltype(glDeleteQueriesEXT)*)eglGetProcAddress("glDeleteQueriesEXT");
decltype(glGetQueryObjectuivEXT) *glGetQueryObjectuivEXT1 = (decltype(glGetQueryObjectuivEXT)*)eglGetProcAddress("glGetQueryObjectuivEXT");
decltype(glGetQueryObjectui64vEXT) *glGetQueryObjectui64vEXT1 = (decltype(glGetQueryObjectui64vEXT)*)eglGetProcAddress("glGetQueryObjectui64vEXT");
decltype(glQueryCounterEXT) *glQueryCounterEXT1 = (decltype(glQueryCounterEXT)*)eglGetProcAddress("glQueryCounterEXT");

```

### 查询耗时

```cpp

GLuint q1;
glGetError();
glGenQueriesEXT1(1, &q1);
glBeginQueryEXT1(GL_TIME_ELAPSED_EXT, q1);

/**
 * put you draw call here
 */

glEndQueryEXT1(GL_TIME_ELAPSED_EXT);

// 这一步不是必须的
glGetQueryObjectuivEXT1(q1, GL_QUERY_RESULT_AVAILABLE_EXT, &avail);
if(avail != GL_TRUE) {
    FUN_ERROR("not avail");
}

// 这一步不是必须的，因为这一步会阻塞等结果, 但如果你不想等，可以用让一句去poll，有数据了再读
glGetQueryObjectuivEXT1(q1, GL_QUERY_RESULT_EXT, &world_time);

glDeleteQueriesEXT1(1, &q1);
```

### 查询时间戳

```cpp
GLuint q2[2];
GLuint64 ts[2];

glGetError(); // clear errors

glGenQueriesEXT1(2, q2);

glQueryCounterEXT1(GL_TIMESTAMP_EXT, q2[0]);
if(auto err = glGetError(); err != GL_NO_ERROR) {
    FUN_ERROR("tsts: error %d", err); // 高通adreno 645似乎还不支持这个API，所以这里返回1280, invalid enum.
}


/**
    * draw call
    */

GLuint avail;
glGetQueryObjectuivEXT1(q2[0], GL_QUERY_RESULT_AVAILABLE_EXT, &avail);
if(avail != GL_TRUE) {
    FUN_ERROR("fps not avail");
}
glGetQueryObjectuivEXT1(q2[1], GL_QUERY_RESULT_AVAILABLE_EXT, &avail);
if(avail != GL_TRUE) {
    FUN_ERROR("fps not avail");
}

glGetQueryObjectui64vEXT1(q2[0], GL_QUERY_RESULT_EXT, &ts[0]);
glGetQueryObjectui64vEXT1(q2[1], GL_QUERY_RESULT_EXT, &ts[1]);

glDeleteQueriesEXT1(2, q2);

FUN_INFO("tsts %lu %lu %lu", ts[1], ts[0], ts[1] - ts[0]);

```

## 注意事项

>  :warning: `adreno 645`在查询GLES extensions的时候能查到GL_EXT_disjoint_timer_query，但是似乎支持的并不完整。查TIME STAMP不work。