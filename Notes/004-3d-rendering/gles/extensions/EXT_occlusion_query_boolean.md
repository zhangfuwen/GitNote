# EXT_occlusion_query_boolean

Name

    EXT_occlusion_query_boolean

Name Strings

    GL_EXT_occlusion_query_boolean

Contributors

    All those who have contributed to the definition of occlusion
    query functionality in the OpenGL ARB and OpenGL ES workgroups,
    upon which this extension spec is entirely dependent.

Contact

    Benj Lipchak, Apple (lipchak 'at' apple.com)

Status

    Complete

Version

    Date: July 22, 2011
    Revision: 2

Number

    OpenGL ES Extension #100

Dependencies

    Written based on the wording of the OpenGL ES 2.0.25 Full Specification
    (November 2, 2010).

Overview

    This extension defines a mechanism whereby an application can
    query whether any pixels (or, more precisely, samples) are drawn
    by a primitive or group of primitives.

    The primary purpose of such a query (hereafter referred to as an
    "occlusion query") is to determine the visibility of an object.
    Typically, the application will render the major occluders in the
    scene, then perform an occlusion query for each detail object in
    the scene. On subsequent frames, the previous results of the
    occlusion queries can be used to decide whether to draw an object
    or not.

New Procedures and Functions

    void GenQueriesEXT(sizei n, uint *ids);
    void DeleteQueriesEXT(sizei n, const uint *ids);
    boolean IsQueryEXT(uint id);
    void BeginQueryEXT(enum target, uint id);
    void EndQueryEXT(enum target);
    void GetQueryivEXT(enum target, enum pname, int *params);
    void GetQueryObjectuivEXT(uint id, enum pname, uint *params);

New Tokens

    Accepted by the <target> parameter of BeginQueryEXT, EndQueryEXT,
    and GetQueryivEXT:

        ANY_SAMPLES_PASSED_EXT                         0x8C2F
        ANY_SAMPLES_PASSED_CONSERVATIVE_EXT            0x8D6A

    Accepted by the <pname> parameter of GetQueryivEXT:

        CURRENT_QUERY_EXT                              0x8865

    Accepted by the <pname> parameter of GetQueryObjectivEXT and
    GetQueryObjectuivEXT:

        QUERY_RESULT_EXT                               0x8866
        QUERY_RESULT_AVAILABLE_EXT                     0x8867

Additions to Chapter 2 of the OpenGL ES 2.0 Specification (OpenGL ES Operation)

    Add a new section "Asynchronous Queries" between sections 2.12 and 2.13
    and renumber subsequent sections:
    
    "2.13 Asynchronous Queries
    
    Asynchronous queries provide a mechanism to return information about the 
    processing of a sequence of GL commands. There is one query type 
    supported by the GL. Occlusion queries (see section 4.1.6) set a boolean 
    to true when any fragments or samples pass the depth test.
    
    The results of asynchronous queries are not returned by the GL immediately
    after the completion of the last command in the set; subsequent commands 
    can be processed while the query results are not complete. When available,
    the query results are stored in an associated query object. The commands 
    described in section 6.1.6 provide mechanisms to determine when query 
    results are available and return the actual results of the query. The 
    name space for query objects is the unsigned integers, with zero reserved 
    by the GL.
    
    Each type of query supported by the GL has an active query object name. If 
    the active query object name for a query type is non-zero, the GL is 
    currently tracking the information corresponding to that query type and the
    query results will be written into the corresponding query object. If the 
    active query object for a query type name is zero, no such information is 
    being tracked.
    
    A query object is created and made active by calling 
    
        void BeginQueryEXT(enum target, uint id);
        
    <target> indicates the type of query to be performed; valid values of 
    <target> are defined in subsequent sections. If the query object name <id> 
    has not been created, the name is marked as used and associated with a new 
    query object of the type specified by <target>. Otherwise <id> must be the 
    name of an existing query object of that type.
    
    BeginQueryEXT fails and an INVALID_OPERATION error is generated if <id> 
    is not a name returned from a previous call to GenQueriesEXT, or if such 
    a name has since been deleted with DeleteQueriesEXT.
    
    BeginQueryEXT sets the active query object name for the query type given 
    by <target> to <id>. If BeginQueryEXT is called with an <id> of zero, if 
    the active query object name for <target> is non-zero (for the targets 
    ANY_SAMPLES_PASSED_EXT and ANY_SAMPLES_PASSED_CONSERVATIVE_EXT, if the 
    active query for either target is non-zero), if <id> is the name of an 
    existing query object whose type does not match <target>, or if <id> is the
    active query object name for any query type, the error INVALID_OPERATION is
    generated.
    
    The command 
    
        void EndQueryEXT(enum target);
        
    marks the end of the sequence of commands to be tracked for the query type 
    given by <target>. The active query object for <target> is updated to 
    indicate that query results are not available, and the active query object 
    name for <target> is reset to zero. When the commands issued prior to 
    EndQueryEXT have completed and a final query result is available, the 
    query object active when EndQueryEXT is called is updated by the GL. The 
    query object is updated to indicate that the query results are available 
    and to contain the query result. If the active query object name for 
    <target> is zero when EndQueryEXT is called, the error INVALID_OPERATION 
    is generated.
    
    The command
    
        void GenQueriesEXT(sizei n, uint *ids);
        
    returns <n> previously unused query object names in <ids>. These names are 
    marked as used, for the purposes of GenQueriesEXT only, but no object is 
    associated with them until the first time they are used by BeginQueryEXT.
    
    Query objects are deleted by calling
    
        void DeleteQueriesEXT(sizei n, const uint *ids);
        
    <ids> contains <n> names of query objects to be deleted. After a query 
    object is deleted, its name is again unused. Unused names in <ids> are 
    silently ignored, as is the value zero. If an active query object is 
    deleted its name immediately becomes unused, but the underlying object is 
    not deleted until it is no longer active (see section C.1).
    
    Query objects contain two pieces of state: a single bit indicating whether 
    a query result is available, and an integer containing the query result 
    value. The number of bits used to represent the query result is 
    implementation-dependent and may vary by query object type. In the initial 
    state of a query object, the result is available and its value is zero.
    
    The necessary state for each query type is an unsigned integer holding the 
    active query object name (zero if no query object is active), and any state
    necessary to keep the current results of an asynchronous query in progress.
    Only a single type of occlusion query can be active at one time, so the 
    required state for occlusion queries is shared."

Additions to Chapter 3 of the OpenGL ES 2.0 Specification (Rasterization)

    None

Additions to Chapter 4 of the OpenGL ES 2.0 Specification (Per-Fragment
Operations and the Frame Buffer)

    Add a new section "Occlusion Queries" between sections 4.1.5 and
    4.1.6 and renumber subsequent sections:

    "4.1.6  Occlusion Queries

    Occlusion queries use query objects to track the number of fragments or 
    samples that pass the depth test. An occlusion query can be started and 
    finished by calling BeginQueryEXT and EndQueryEXT, respectively, with a 
    target of ANY_SAMPLES_PASSED_EXT or ANY_SAMPLES_PASSED_CONSERVATIVE_EXT.

    When an occlusion query is started with the target 
    ANY_SAMPLES_PASSED_EXT, the samples-boolean state maintained by the GL is
    set to FALSE. While that occlusion query is active, the samples-boolean 
    state is set to TRUE if any fragment or sample passes the depth test. When 
    the occlusion query finishes, the samples-boolean state of FALSE or TRUE is
    written to the corresponding query object as the query result value, and 
    the query result for that object is marked as available. If the target of 
    the query is ANY_SAMPLES_PASSED_CONSERVATIVE_EXT, an implementation may 
    choose to use a less precise version of the test which can additionally set
    the samples-boolean state to TRUE in some other implementation dependent 
    cases."

Additions to Chapter 5 of the OpenGL ES 2.0 Specification (Special Functions)

    None

Additions to Chapter 6 of the OpenGL ES 2.0 Specification (State and
State Requests)

    Add a new section "Asynchronous Queries" between sections 6.1.5 and
    6.1.6 and renumber subsequent sections:

    "6.1.6  Asynchronous Queries

    The command

      boolean IsQueryEXT(uint id);

    returns TRUE if <id> is the name of a query object.  If <id> is zero,
    or if <id> is a non-zero value that is not the name of a query
    object, IsQueryEXT returns FALSE.

    Information about a query target can be queried with the command

      void GetQueryivEXT(enum target, enum pname, int *params);

    <target> identifies the query target, and must be one of 
    ANY_SAMPLES_PASSED_EXT or ANY_SAMPLES_PASSED_CONSERVATIVE_EXT.
    
    <pname> must be CURRENT_QUERY_EXT. The name of the currently active
    query for <target>, or zero if no query is active, will be placed in
    <params>.

    The state of a query object can be queried with the command

      void GetQueryObjectuivEXT(uint id, enum pname, uint *params);

    If <id> is not the name of a query object, or if the query object
    named by <id> is currently active, then an INVALID_OPERATION error is
    generated. <pname> must be QUERY_RESULT_EXT or QUERY_RESULT_AVAILABLE_EXT.

    If <pname> is QUERY_RESULT_EXT, then the query object's result value
    is returned as a single integer in <params>. If the value is so large in 
    magnitude that it cannot be represented with the requested type, then the 
    nearest value representable using the requested type is returned.

    There may be an indeterminate delay before the above query returns. If 
    <pname> is QUERY_RESULT_AVAILABLE_EXT, FALSE is returned if such a delay 
    would be required; otherwise TRUE is returned. It must always be true that 
    if any query object returns a result available of TRUE, all queries of the 
    same type issued prior to that query must also return TRUE.

    Querying the state for any given query object forces that occlusion query 
    to complete within a finite amount of time. Repeatedly querying the 
    QUERY_RESULT_AVAILABLE_EXT state for any given query object is guaranteed
    to return true eventually. Note that multiple queries to the same occlusion
    object may result in a significant performance loss. For better performance
    it is recommended to wait N frames before querying this state. N is 
    implementation dependent but is generally between one and three.
    
    If multiple queries are issued using the same object name prior to calling 
    GetQueryObjectuivEXT, the result and availability information returned 
    will always be from the last query issued. The results from any queries 
    before the last one will be lost if they are not retrieved before starting 
    a new query on the same <target> and <id>."

Additions to Appendix C of the OpenGL ES 2.0 Specification (Shared Object and 
Multiple Contexts)

    Change the first sentence of the first paragraph of section C.1.2 to read:
    
    "When a buffer, texture, renderbuffer, or query is deleted, ..."
 
    Add the following sentence to the end of section C.1.2:

    "A query object is in use so long as it is the active query object for a 
    query type and index, as described in section 2.13."

Errors

    The error INVALID_OPERATION is generated if BeginQueryEXT is called
    where <id> is not a name returned from a previous call to GenQueriesEXT,
    or if such a name has since been deleted with DeleteQueriesEXT.
    
    The error INVALID_OPERATION is generated if BeginQueryEXT is called
    where <id> is zero.
    
    The error INVALID_OPERATION is generated if BeginQueryEXT is called
    where <id> is the name of an existing query object whose type does not 
    match <target>.
    
    The error INVALID_OPERATION is generated if BeginQueryEXT is called
    where <id> is the active query object name for any query type.
    
    The error INVALID_OPERATION is generated if BeginQueryEXT is called
    when the active query object name for either ANY_SAMPLES_PASSED_EXT or
    ANY_SAMPLES_PASSED_CONSERVATIVE_EXT is non-zero.

    The error INVALID_OPERATION is generated if EndQueryEXT is called
    when the active query object name for <target> is zero.
    
    The error INVALID_OPERATION is generated if GetQueryObjectuivEXT is 
    called where <id> is not the name of a query object.

    The error INVALID_OPERATION is generated if GetQueryObjectuivEXT is 
    called where <id> is the name of a currently active query object.

    The error INVALID_VALUE is generated if GenQueriesEXT is called where
    <n> is negative.

    The error INVALID_VALUE is generated if DeleteQueriesEXT is called
    where <n> is negative.

    The error INVALID_ENUM is generated if BeginQueryEXT, EndQueryEXT,
    or GetQueryivEXT is called where <target> is not
    ANY_SAMPLES_PASSED_EXT or ANY_SAMPLES_PASSED_CONSERVATIVE_EXT.

    The error INVALID_ENUM is generated if GetQueryivEXT is called where
    <pname> is not CURRENT_QUERY_EXT.

    The error INVALID_ENUM is generated if GetQueryObjectuivEXT is called 
    where <pname> is not QUERY_RESULT_EXT or QUERY_RESULT_AVAILABLE_EXT.

New State

(table 6.18, p. 233)

                                                            Int'l
    Get Value                   Type  Get Command           Value  Description             Sec
    --------------------------  ----  --------------------  -----  ----------------------  -----
    -                           B     -                     FALSE  query active            4.1.6
    CURRENT_QUERY_EXT           Z+    GetQueryivEXT         0      active query ID         4.1.6
    QUERY_RESULT_EXT            B     GetQueryObjectuivEXT  FALSE  samples-passed          4.1.6
    QUERY_RESULT_AVAILABLE_EXT  B     GetQueryObjectuivEXT  FALSE  query result available  4.1.6

Issues

    (1)  What should the enum be called?

        RESOLVED: The enum should be called ANY_SAMPLES_PASSED as in
        ARB_occlusion_query2 to retain compatibility between the two
        extensions.

    (2)  Can application-provided names be used as query object names?

        ARB_occlusion_query allows application-provided names, but this
        was later removed in core OpenGL.

        RESOLVED: No, we will follow core OpenGL on this.

    (3)  Should calling GenQueries or DeleteQueries when a query is
         active produce an error?

        This behavior is in ARB_occlusion_query but was
        removed in OpenGL 3.0.

        RESOLVED: Not an error.  Calling DeleteQueries marks the name
        as no longer used, but the object is not deleted until it is no 
        longer in use (i.e. no longer active).

    (4)  What is the interaction with multisample?

        RESOLVED: The query result is set to true if at least one
        sample passes the depth test.

    (5)  Exactly what stage in the pipeline are we counting samples at?

        RESOLVED: We are counting immediately after _both_ the depth and
        stencil tests, i.e., samples that pass both.  Note that the depth
        test comes after the stencil test, so to say that it is the
        number that pass the depth test is sufficient; though it is often
        conceptually helpful to think of the depth and stencil tests as
        being combined, because the depth test's result impacts the
        stencil operation used.

    (6)  Is it guaranteed that occlusion queries return in order?

        RESOLVED: Yes.

        If occlusion test X occurred before occlusion query Y, and the driver 
        informs the app that occlusion query Y is done, the app can infer that 
        occlusion query X is also done.

    (7) Will polling a query for QUERY_RESULT_AVAILABLE without a Flush
        possibly cause an infinite loop?

        RESOLVED: No.

    (8) Should there be a "target" parameter to BeginQuery?

        RESOLVED: Yes.  This distinguishes the boolean queries
        defined by this extension (and ARB_occlusion_query2) from
        the counter queries defined by ARB_occlusion_query.

    (9) Are query objects shareable between multiple contexts?

        RESOLVED: No.  Query objects are lightweight and we normally share 
        large data across contexts.  Also, being able to share query objects
        across contexts is not particularly useful.  In order to do the async 
        query across contexts, a query on one context would have to be finished 
        before the other context could query it.  

    (10) Should there be a limit on how many queries can be outstanding?

        RESOLVED: No. If an implementation has an internal limit, it can
        flush the pipeline when it runs out.

    (11) Can an implementation sometimes return a conservative result,
         i.e. return true even though no samples were drawn?

        RESOLVED: Yes, but only when explicitly enabled by the
        application.

        Allowing such results with no restrictions effectively makes
        the functionality of the extension optional, which decreases
        its value. Precise restrictions are presumably hard to
        specify.

        One situation where this restriction could be relevant is if
        an implementation performs a conservative early depth test at
        a lower precision and wants to base the occlusion query result
        on that whenever the early depth test can be used.

    (12) Should the restrictions in issue 11 be explicitly enabled
         by the application in order to be in effect?

        RESOLVED: Yes.

        The restrictions could be enabled by a hint call or by using
        a different enum in the BeginQuery call.

        This would enable the application to choose whether it wants a
        precise (but possibly slow) version or an approximate (but
        possibly faster) version.

    (13) Can the restrictions in issue 18 be applied nondeterministically?

        An implementation might benefit from taking the decision of
        whether to apply a particular restriction on a case by case
        basis. Some of these decisions could depend on
        nondeterministic effects such as memory bus timing.

        RESOLVED: No. This would violate the GL repeatability
        principle.

    (14) How does an application request that the result is allowed to
         be conservative (as defined in issue 11)?

        RESOLVED: It is specified as a separate query target,
        ANY_SAMPLES_PASSED_CONSERVATIVE.


Revision History

   Date: 5/03/2011
   Revision: 1 (Benj Lipchak)
      - Initial draft based on XXX_occlusion_query_boolean

   Date: 7/22/2011
   Revision: 2 (Benj Lipchak)
      - Rename from APPLE to EXT
