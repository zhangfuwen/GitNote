# INTEL_performance_query

Name

    INTEL_performance_query

Name Strings

    GL_INTEL_performance_query

Contact

   Tomasz Madajczak, Intel (tomasz.madajczak 'at' intel.com)

Contributors

    Piotr Uminski, Intel
    Slawomir Grajewski, Intel

Status

    Complete, shipping on selected Intel graphics.

Version

    Last Modified Date: December 20, 2013
    Revision: 3

Number

    OpenGL Extension #443
    OpenGL ES Extension #164

Dependencies

    OpenGL dependencies:

        OpenGL 3.0 is required.

        The extension is written against the OpenGL 4.4 Specification, Core
        Profile, October 18, 2013.

    OpenGL ES dependencies:

        This extension is written against the OpenGL ES 2.0.25 Specification
        and OpenGL ES 3.0.2 Specification.

Overview

    The purpose of this extension is to expose Intel proprietary hardware
    performance counters to the OpenGL applications. Performance counters may
    count:

    - number of hardware events such as number of spawned vertex shaders. In
      this case the results represent the number of events.

    - duration of certain activity, like time took by all fragment shader
      invocations. In that case the result usually represents the number of
      clocks in which the particular HW unit was busy. In order to use such
      counter efficiently, it should be normalized to the range of <0,1> by
      dividing its value by the number of render clocks.

    - used throughput of certain memory types such as texture memory. In that
      case the result of performance counter usually represents the number of
      bytes transferred between GPU and memory.

    This extension specifies universal API to manage performance counters on
    different Intel hardware platforms. Performance counters are grouped
    together into proprietary, hardware-specific, fixed sets of counters that
    are measured together by the GPU.

    It is assumed that performance counters are started and ended on any
    arbitrary boundaries during rendering. 

    A set of performance counters is represented by a unique query type. Each
    query type is identified by assigned name and ID. Multiple query types
    (sets of performance counters) are supported by the Intel hardware. However
    each Intel hardware generation supports different sets of performance
    counters.  Therefore the query types between hardware generations can be
    different. The definition of query types and their results structures can
    be learned through the API. It is also documented in a separate document of
    Intel OGL Performance Counters Specification issued per each new hardware
    generation.

    The API allows to create multiple instances of any query type and to sample
    different fragments of 3D rendering with such instances. Query instances
    are identified with handles.

New Procedures and Functions

    void GetFirstPerfQueryIdINTEL(uint *queryId);

    void GetNextPerfQueryIdINTEL(uint queryId, uint *nextQueryId);

    void GetPerfQueryIdByNameINTEL(char *queryName, uint *queryId);

    void GetPerfQueryInfoINTEL(uint queryId,
             uint queryNameLength, char *queryName,
             uint *dataSize, uint *noCounters, 
             uint *noInstances, uint *capsMask);

    void GetPerfCounterInfoINTEL(uint queryId, uint counterId,
             uint counterNameLength, char *counterName,
             uint counterDescLength, char *counterDesc,
             uint *counterOffset, uint *counterDataSize, uint *counterTypeEnum,
             uint *counterDataTypeEnum, uint64 *rawCounterMaxValue);

    void CreatePerfQueryINTEL(uint queryId, uint *queryHandle);

    void DeletePerfQueryINTEL(uint queryHandle);

    void BeginPerfQueryINTEL(uint queryHandle);

    void EndPerfQueryINTEL(uint queryHandle);

    void GetPerfQueryDataINTEL(uint queryHandle, uint flags,
             sizei dataSize, void *data, uint *bytesWritten);

New Tokens

    Returned by the capsMask parameter of GetPerfQueryInfoINTEL

        PERFQUERY_SINGLE_CONTEXT_INTEL          0x0000
        PERFQUERY_GLOBAL_CONTEXT_INTEL          0x0001

    Accepted by the flags parameter of GetPerfQueryDataINTEL

        PERFQUERY_WAIT_INTEL                    0x83FB
        PERFQUERY_FLUSH_INTEL                   0x83FA
        PERFQUERY_DONOT_FLUSH_INTEL             0x83F9

    Returned by GetPerfCounterInfoINTEL function as counter type enumeration in
    location pointed by counterTypeEnum

        PERFQUERY_COUNTER_EVENT_INTEL           0x94F0
        PERFQUERY_COUNTER_DURATION_NORM_INTEL   0x94F1
        PERFQUERY_COUNTER_DURATION_RAW_INTEL    0x94F2
        PERFQUERY_COUNTER_THROUGHPUT_INTEL      0x94F3
        PERFQUERY_COUNTER_RAW_INTEL             0x94F4
        PERFQUERY_COUNTER_TIMESTAMP_INTEL       0x94F5

    Returned by glGetPerfCounterInfoINTEL function as counter data type
    enumeration in location pointed by counterDataTypeEnum

        PERFQUERY_COUNTER_DATA_UINT32_INTEL     0x94F8
        PERFQUERY_COUNTER_DATA_UINT64_INTEL     0x94F9
        PERFQUERY_COUNTER_DATA_FLOAT_INTEL      0x94FA
        PERFQUERY_COUNTER_DATA_DOUBLE_INTEL     0x94FB
        PERFQUERY_COUNTER_DATA_BOOL32_INTEL     0x94FC

   Accepted by the <pname> parameter of GetIntegerv:

        PERFQUERY_QUERY_NAME_LENGTH_MAX_INTEL   0x94FD
        PERFQUERY_COUNTER_NAME_LENGTH_MAX_INTEL 0x94FE
        PERFQUERY_COUNTER_DESC_LENGTH_MAX_INTEL 0x94FF

    Accepted by the <pname> parameter of GetBooleanv:

        PERFQUERY_GPA_EXTENDED_COUNTERS_INTEL   0x9500

Add new Section 4.4 to Chapter 4, Event Model for OpenGL 4.4
Add new Section 2.18 to Chapter 2, OpenGL ES Operation for OpenGL ES 3.0.2

    4.4 Performance Queries (for OpenGL 4.4)
    2.18 Performance Queries (for OpenGL ES 3.0.2)

    Hardware and software performance counters can be used to obtain
    information about GPU activity. Performance counters are grouped into query
    types. Different query types can be supported on different hardware
    platforms and/or driver versions. One or more instances of the query types
    can be created.

    Each query type has unique query ID. Query ids supported on given platform
    can be queried in the run-time. Function:

        void GetFirstPerfQueryIdINTEL(uint *queryId);

    returns the identifier of the first performance query type that is
    supported on a given platform. The result is passed in location pointed by
    queryId parameter. If the given hardware platform doesn't support any
    performance queries, then the value of 0 is returned and INVALID_OPERATION
    error is raised. If queryId pointer is equal to 0, INVALID_VALUE error is
    generated.

    Next query ids can be queried by multiply call to the function:

        void GetNextPerfQueryIdINTEL(uint queryId, uint *nextQueryId);

    This function returns the integer identifier of the next performance query
    on a given platform to the specified with queryId. The result is passed in
    location pointed by nextQueryId. If query identified by queryId is the last
    query available the value of 0 is returned. If the specified performance
    query identifier is invalid then INVALID_VALUE error is generated. If
    nextQueryId pointer is equal to 0, an INVALID_VALUE error is
    generated. Whenever error is generated, the value of 0 is returned.

    Each performance query type has a name and a unique identifier. The query
    identifier for a given query name be read using function:

        void GetPerfQueryIdByNameINTEL(char *queryName, uint *queryId);

    This function returns the identified of the query type specified by the
    string provided as queryName parameter.  If queryName does not reference a
    valid query name, an INVALID_VALUE error is generated.

    General description of a query type can be read using the function:

        void GetPerfQueryInfoINTEL(uint queryId, uint queryNameLength,
            char *queryName, uint *dataSize,
            uint *noCounters, uint *maxInstances,
            uint *noActiveInstances, uint *capsMask);

    The function returns information about the performance query specified with
    queryId parameter, particularly:

    -  query name in queryName location. The maximal name is specified by
       queryNameLength

    -  size of query output structure in bytes in dataSize location

    -  number of performance counters in the query output structure in
       noCounters location

    -  the maximal allowed number of query instances that can be created on a
       given architecture in maxInstances location. Because the other type queries
       are created using the same resources, it may happen that the actual amount
       of created instances is smaller than the returned number

    -  the actual number of already created query instances in maxInstances
       location

    -  mask of query capabilities in capsMask location.

    If the mask returned in capsMask contains PERFQUERY_SINGLE_CONTEXT_INTEL
    token this means the query supports context sensitive measurements,
    otherwise, if the mask contains token of GL_PERFQUERY_GLOBAL_CONTEXT_INTEL
    this means the query doesn't support that feature and the counters will be
    updated for all render contexts as they are global for hardware.

    If queryId does not reference a valid query type, an INVALID_VALUE error is
    generated.

    Performance counters that belong to the same query type have unique
    ids. Performance counter ids values start with 1. Performance counter id 0
    is reserved as an invalid counter. Information about performance counters
    that belongs to a given query type can be read using the function:

    void GetPerfCounterInfoINTEL(uint queryId, uint counterId, 
         uint counterNameLength, char *counterName, 
         uint counterDescLength, char *counterDesc, 
         uint *counterOffset, uint *counterDataSize, uint *counterTypeEnum,
         uint *counterDataTypeEnum, uint64 *rawCounterMaxValue);

    The function returns descriptive information about each particular
    performance counter that is an element of the performance query. The
    counter is identified with a pair of queryId and counterId parameters. The
    following parameters are returned:

    -  counter name in counterName location. The maximal length of copied name
       is specified with counterNameLength.

    -  counter description text in  counterDesc location. The maximal length of
       copied text is specified with counterDescLength.

    -  byte offset of the counter from the start of the query structure in
       counterOffset location.

    -  counter size in bytes in  counterDataSize location.

    -  counter type enumeration in counterTypeEnum location. It can be one o
       the following tokens:
           PERFQUERY_COUNTER_EVENT_INTEL
           PERFQUERY_COUNTER_DURATION_NORM_INTEL
           PERFQUERY_COUNTER_DURATION_RAW_INTEL
           PERFQUERY_COUNTER_THROUGHPUT_INTEL
           PERFQUERY_COUNTER_RAW_INTEL
           PERFQUERY_COUNTER_TIMESTAMP_INTEL

    -  counter data type enumeration, in counterDataTypeEnum location. It can
       be one o the following tokens:
           PERFQUERY_COUNTER_DATA_UINT32_INTEL
           PERFQUERY_COUNTER_DATA_UINT64_INTEL
           PERFQUERY_COUNTER_DATA_FLOAT_INTEL
           PERFQUERY_COUNTER_DATA_DOUBLE_INTEL
           PERFQUERY_COUNTER_DATA_BOOL32_INTEL

    -  for some raw counters for which the maximal value is deterministic, the
       maximal value of the counter in 1 second is returned in the location
       pointed by rawCounterMaxValue, otherwise, the location is written with
       the value of 0.

    If the pair of queryId and counterId does not reference a valid counter,
    an INVALID_VALUE error is generated.

    A single instance of the performance query of a given type can be created
    using function:

        void CreatePerfQueryINTEL(uint queryId, uint *queryHandle);

    The handle to newly created query instance is returned in queryHandle
    location. If queryId does not reference a valid query type,
    an INVALID_VALUE error is generated. If the query instance cannot be
    created due to exceeding the number of allowed instances or driver fails
    query creation due to an insufficient memory reason, an OUT_OF_MEMORY error
    is generated, and the location pointed by queryHandle returns NULL.
    Existing query instance can be deleted using function

        void DeletePerfQueryINTEL(uint queryHandle);

    queryHandle must be a query instance handle returned by
    CreatePerfQueryINTEL(). If a query handle doesn't reference a previously
    created performance query instance, an INVALID_VALUE error is generated.

    A new measurement session for a given query instance can be started using
    function:

        void BeginPerfQueryINTEL(uint queryHandle);

    where queryHandle must be a query instance handle returned by
    CreatePerfQueryINTEL(). If a query handle doesn't reference a previously
    created performance query instance, an INVALID_VALUE error is
    generated. Note that some query types, they cannot be collected in the same
    time. Therefore calls of BeginPerfQueryINTEL() cannot be nested if they
    refer to queries of such different types. In such case INVALID_OPERATION
    error is generated.

    The counters may not start immediately after BeginPerfQueryINTEL().
    Because the API and GPU are asynchronous, the start of performance counters
    is delayed until the graphics hardware actually executes the hardware
    commands issued by this function.  However, it is guaranteed that collecting
    of performance counters will start before any draw calls specified in the
    same context after call to BeginPerfQueryINTEL().

    Collecting performance counters may be stopped by a function:

        void EndPerfQueryINTEL(uint queryHandle);

    where queryHandle must be a query instance handle returned by
    CreatePerfQueryINTEL(). The function ends the measurement session started
    by BeginPerfQueryINTEL().  If a performance query is not currently started,
    an INVALID_OPERATION error will be generated. Similarly as in
    glBeginPerfQueryINTEL() case, the execution of glEndPerfQueryINTEL() is not
    immediate. The end of measurement is delayed until graphics hardware
    completes processing of the hardware commands issued by this
    function. However, it is guaranteed that results any draw calls specified in
    the same context after call to EndPerfQueryINTEL() will be not measured by
    this query.

    The query result can be read using function:

        void GetPerfQueryDataINTEL(uint queryHandle, uint flags, sizei
            dataSize, void *data, uint *bytesWritten);

    The function returns the values of counters which have been measured within
    the query session identified by queryHandle.  The call may end without
    returning any data if they are not ready for reading as the measurement
    session is still pending (the EndPerfQueryINTEL() command processing is not
    finished by hardware). In this case location pointed by the bytesWritten
    parameter will be set to 0. The meaning of the flags parameter is the
    following:

    -  PERFQUERY_DONOT_FLUSH_INTEL means that the call of
       GetPerfQueryDataINTEL() is non-blocking, which checks for results and
       returns them if they are available. Otherwise, (if the results of the
       query are not ready) it returns without flushing any outstanding 3D
       commands  to the GPU. The use case for this is when a flush of
       outstanding 3D commands to GPU has already been ensured with other
       OpenGL API calls. 
 
    -  PERFQUERY_FLUSH_INTEL means that the call of GetPerfQueryDataINTEL() is
       non-blocking, which checks for results and returns them if they are
       available. Otherwise, it implicitly submits any outstanding 3D commands
       to the GPU for execution. In that case the subsequent call of
       glGetPerfQueryDataINTEL() may return data once the query completes.

    -  PERFQUERY_WAIT_INTEL means that the call of GetPerfQueryDataINTEL() is
       blocking and waits till the query results are available and returns
       them. It means that if the query results are not yet available then it
       implicitly submits any outstanding 3D commands to GPU and waits for the
       query completion.

    If the measurement session indentified by queryHandle is completed then the
    call of GetPerfQueryDataINTEL() always writes query result to the location
    pointed by the data parameter and the amount of bytes written is stored in
    the location pointed by the bytesWritten parameter.

    If bytesWritten or data pointers are NULL then an INVALID_VALUE error is
    generated.


New Implementation Dependent State

Add new Table 23.75 to Chapter 23, State Tables (OpenGL 4.4)
Add new Table 6.37 to Chapter 6.2, State Tables (OpenGL ES 3.0.2)


    Get Value                              Type Get Command Value Description
    ------------------------------         ---- ----------- ----- -------------
    PERFQUERY_QUERY_NAME_LENGTH_MAX_INTEL   Z+ GetIntegerv  256   max query name length
    PERFQUERY_COUNTER_NAME_LENGTH_MAX_INTEL Z+ GetIntegerv  256   max counter name length
    PERFQUERY_COUNTER_DESC_LENGTH_MAX_INTEL Z+ GetIntegerv  1024  max description length
    PERFQUERY_GPA_EXTENDED_COUNTERS_INTEL   B  GetBooleanv  -     extended counters available


Issues

    1. What is the usage model of this extension?

    Generally there are two approaches of measuring performance with Intel OGL
    Performance Queries, such as:

    - Per draw call measurements - performance counters can be used to assess
      the business of particular 3D hardware units under assumption that 3D
      hardware is almost 100% time busy from the CPU point of view.

    - Per 3D scene measurements - performance counters can be used to assess
      the balance of CPU and GPU processing times. Such assessment shows whether
      the workload is CPU whether GPU bound.

    2. How per draw call measurements are performed?

       In the per-draw call usage model each call to the draw routine
       (e.g. glDrawArrays, glDrawElements) should be surrounded by a dedicated
       query instance. That means that each draw operation should be measured
       independently. It is recommended to measure the GPU performance
       characteristics for a single draw call to find possible bottlenecks
       for the application executed on a given hardware.

    3. How per scene measurements are performed?

       The usage model assumes that one performance query instance measures a
       complete scene. It is recommended to figure out if the workload is CPU
       or GPU bound. It should be noted that:

       - For a longer scope of performance query the probability of 3D hardware
         frequency change is higher. The higher probability of frequency change
         causes that the larger percentage of results may be biased with gross
         errors. 

       - For complicated 3D scenes the condition of render commands split is
         always met. 

       Thus, to calculate an average 3D hardware unit utilization for a longer
       period of time it is recommended to use a larger number of per draw call
       queries rather than a lower number of per 3D scene queries. It is
       recommended to use this method when application uses full screen mode as
       current implementation of queries supports only global context.

    4. How results of the query can be read?

       Results of the queries cannot be read before the entire drawing is done
       by the GPU. This means that the application programmer has to decide
       about the synchronization method it uses to read the query
       results. There are the following options:

       - Use glFlush to trigger submission of any pending commands to the
         GPU. Later check results availability with repetitive non-blocking
         calls to GetPerfQueryDataINTEL function using the synchronization flag
         of GL_PERFQUERY_DONOT_FLUSH_INTEL. 

       - Use flag GL_PERFQUERY_FLUSH_INTEL in glGetPerfQueryDataINTEL to
         trigger submission of any pending commands to the GPU. If results are
         not immediately available, check their availability with repetitive
         non-blocking calls to GetPerfQueryDataINTEL function using the
         synchronization flag of GL_PERFQUERY_DONOT_FLUSH_INTEL.

       - Do a blocking call to glGetPerfQueryDataINTEL() with
         GL_PERFQUERY_WAIT_INTEL flag set. The flag ensures that any pending GPU
         commands are submitted and function blocks till GPU results are
         available.

       It is allowed to perform simultaneous measurements with multiple active
       queries of the same type. However it may be not allowed to perform
       simultaneous measurements of queries with different types, as it may
       require reprogramming of the same hardware part and could destroy the
       hardware settings of the previous query.

    5. Are query results always accurate?

       There are certain hardware conditions which may cause the results
       of performance counters expressed in hardware clocks to be inaccurate.
       The conditions may include:

       - Render clock change -  the condition usually causes that all counter
         values expressed in hardware clocks are incorrect. It is indicated by
         FrequencyChanged flag.
         
       - Render commands split - in some cases GPU has to split execution of
         drawing operations surrounded by the query into at least two
         parts. The condition usually causes that counter values expressed in
         time domains (in microseconds) may be substantially larger than the
         average values of that counter. It is indicated by SplitOccured flag.

       - Rendering preemption - if GPU is shared among two or more 3D
         applications, the hardware counters gathered in a global mode contain
         additive results for these applications. The condition is also
         indicated with SplitOccured flag.

       The above conditions are indicated in special fields in the query
       results structures. It is up to the user to decide if the results are to
       be processed further or dropped. In certain cases it can be determined
       that the render commands split condition always occurs and has to be
       accepted.

    6. Are query results per-context or global?

       Some GPU platforms and/or driver versions support only global GPU
       counters. In such cases, the query instance has to have
       GL_PERFQUERY_GLOBAL_CONTEXT_INTEL flag set when creating query
       instance. Otherwise, creation will fail and an INVALID_OPERATION error
       will be generated.

       Support for a global context means that a single query instance measures
       all GPU activities performed between query start and query end. Query
       measures not only current OpenGL context but also activities of other
       OpenGL contexts, other 3D API like DX and operating system windows draw
       calls.

Program examples

    1. Reading counter  meta data example
  
       // query data has proprietary predefined structure layout
       // associated with the vendor query ID
       GL_QUERY_PIPELINE_METRICS * pQueryData;
  
       uint queryId;
       uint nextQueryId;
       uint queryHandle;
       uint dataSize;
       uint noCounters;
       uint noInstances;
       uint capsMask;
  
       const uint queryNameLen = 32;
       char queryName[queryNameLen];
  
       const uint counterNameLen = 32;
       char counterName[counterNameLen];
      
       const uint counterDescLen = 256;
       char counterDesc[counterDescLen];
  
       //get first vendor queryID
       glGetFirstPerfQueryIdINTEL(&queryId);
  
       nextQueryId = queryId;
       while(nextQueryId)
       {
           glGetPerfQueryInfoINTEL(
               nextQueryId,
               queryNameLen,
               &queryName,
               &dataSize,
               &noCounters,
               &noInstances,
               &capsMask);
  
               for(int counterId = 1; counterId <= noCounters; counterId++)
           {
               uint counterOffset;
               uint counterDataSize;
               uint counterTypeEnum;
               uint counterDataTypeEnum;
               UINT64 rawCounterMaxValue;
  
               glGetPerfCounterInfoINTEL(
                   nextQueryId,
                   counterId,
                   counterNameLen,
                   counterName,
                   counterDescLen,
                   counterDesc,
                   &counterOffset,
                   &counterDataSize,
                   &counterTypeEnum,
                   &counterDataTypeEnum,
                   &rawCounterMaxValue);
  
                   // use returned values here
                   ...
           }
       }
  
    2. Measuring a single draw call example
  
       Note that GL_QUERY_PIPELINE_METRICS is a proprietary structure defined
       by vendor and is used as example and function named according to the
       convention of glFuntionINTEL are wrappers to dynamically linked-by-name
       procedures.
  
       // query data has proprietary predefined structure layout
       // associated with the vendor query ID
       GL_QUERY_PIPELINE_METRICS * pQueryData;
  
       uint queryId;
       uint queryHandle;
       char queryName[] = "Intel_Pipeline_Query";
  
       // get vendor queryID by name
       glGetPerfQueryIdByNameINTEL(queryName, &queryId);
  
       // create query instance of queryId type
       glCreatePerfQueryINTEL(queryId, &queryHandle);
  
       glBeginPerfQueryINTEL(queryHandle); // Start query
  
       glDrawElements(...); // Issue graphics commands, do whatever
  
       glEndPerfQueryINTEL(queryHandle); // End query
  
       // perform other application activities
  
       uint bytesWritten = 0;
       uint dataSize = sizeof(GL_QUERY_PIPELINE_METRICS);
      
       pQueryData = (GL_QUERY_PIPELINE_METRICS *) malloc(dataSize);
  
       // for the first time use GL_PERFQUERY_FLUSH_INTEL flag to ensure graphics
       // commands were submitted to hardware
  
       glGetPerfQueryDataINTEL(
           queryHandle,
           GL_PERFQUERY_FLUSH_INTEL,
           dataSize,
           pQueryData,
           &bytesWritten);
  
       while(bytesWritten == 0)
       {
           // Now enough to use GL_PERFQUERY_DONOT_FLUSH_INTEL flag
           glGetPerfQueryDataINTEL(
               queryHandle,
               GL__PERFQUERY_DONOT_FLUSH_INTEL,
                   dataSize,
               pQueryData,
               &bytesWritten);
       }
  
       if(bytesWritten == dataSize)
       {
           // Use counters' data here
           uint64 vertexShaderKernelsRunCount =
                pQueryData->VertexShaderInvocations;
           uint64 fragmentShaderKernelsRunCount =
                pQueryData->FragmentShaderInvocations;
           ...
       }
       else
       {
          // error handling case
       }
  
       glDeletePerfQueryINTEL(queryHandle); // query instance is released
  
    3. Measuring multiple draw calls with synchronous wait for result
  
       Note that GL_QUERY_HD_HW_METRICS is a proprietary structure defined by
       vendor and is used as example and function named according to the
       convention of glFuntionINTEL are wrappers to dynamically linked-by-name
       procedures.
  
       // query data has proprietary predefined structure layout
       // associated with the vendor query ID
       GL_QUERY_HD_HW_METRICS * pQueryData;
  
       uint queryId;
       UINT32 queryHandle[1000];
       char queryName[] = "Intel_HD_Hardware_Counters";
  
       // get vendor queryID by name
       glGetPerfQueryIdByNameINTEL(queryName, &queryId);
  
       // create memory for 1000 results
       uint dataSize = sizeof(GL_QUERY_HD_HW_METRICS);
       pQueryData = (GL_QUERY_HD_HW_METRICS *) malloc(dataSize * 1000);
  
       // create 1000 query instances of queryId type
       for(int i = 0; i < 1000; i++)
       {
           glCreatePerfQueryINTEL(queryId, &queryHandle[i]);
       }
  
       uint currentDrawNumber = 0;
  
       // start 1st query
       glBeginPerfQueryINTEL(queryHandle[currentDrawNumber]);
  
       glDrawElements(...); // Issue graphics commands
  
       // end query
       glEndPerfQueryINTEL(queryHandle[currentDrawNumber++]);
  
       ...
  
       // start nth query
       glBeginPerfQueryINTEL(queryHandle[currentDrawNumber]);
  
       glDrawElements(...); // Issue graphics commands
  
       // end query
       glEndPerfQueryINTEL(queryHandle[currentDrawNumber++]);
  
       ...
  
       // assume currentDrawNumber == 1000 here
       // so get all results after these 1000 draws
  
       GL_QUERY_HD_HW_METRICS *pData = pQueryData;
      
       for(int i = 0; i < 1000; i++)
       {
           uint bytesWritten = 0;
  
           // use GL_PERFQUERY_WAIT_INTEL flag to cause the function will wait
           // for the query completion
           glGetPerfQueryDataINTEL(
               queryHandle[i],
               GL_PERFQUERY_WAIT_INTEL,
               dataSize,
               pData,
               &bytesWritten);
  
           if(bytesWritten != sizeof(GL_QUERY_HD_HW_METRICS))
           {
                // query error case
                assert(false);
                ...
                    // some cleanup needed also
                ...
                return ERROR;
           }
  
           pData++;
        }
  
        // use counters data
        ...
  
        // repeat measurements if needed reusing the query instances
        ...
  
        // query instances are no longer needed so release all of them
        for(int i = 0; i < 1000; i++)
        {
            glDeletePerfQueryINTEL(queryHandle[i]);
        }
  
        return SUCCESS;

Revision History

    1.3   20/12/13 Jon Leech  Assign extension #s and enum values. Fix
                              a few typos (Bug 11345).

    1.2   29/11/13 sgrajewski Extension upgraded to 4.4 core specification.
                              ES3.0.2 dependencies added.

    1.1   06/06/11 puminski   Initial revision.
