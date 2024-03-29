# AMD_performance_monitor

Name

    AMD_performance_monitor
    
Name Strings

    GL_AMD_performance_monitor
    
Contributors

    Dan Ginsburg
    Aaftab Munshi
    Dave Oldcorn
    Maurice Ribble
    Jonathan Zarge

Contact

    Dan Ginsburg (dan.ginsburg 'at' amd.com)

Status

    ???

Version

    Last Modified Date: 11/29/2007

Number

    OpenGL Extension #360
    OpenGL ES Extension #50

Dependencies

    None

Overview

    This extension enables the capture and reporting of performance monitors.
    Performance monitors contain groups of counters which hold arbitrary counted 
    data.  Typically, the counters hold information on performance-related
    counters in the underlying hardware.  The extension is general enough to
    allow the implementation to choose which counters to expose and pick the
    data type and range of the counters.  The extension also allows counting to 
    start and end on arbitrary boundaries during rendering.

Issues

    1.  Should this be an EGL or OpenGL/OpenGL ES extension?

        Decision - Make this an OpenGL/OpenGL ES extension
        
        Reason - We would like to expose this extension in both OpenGL and 
        OpenGL ES which makes EGL an unsuitable choice.  Further, support for 
        EGL is not a requirement and there are platforms that support OpenGL ES 
        but not EGL, making it difficult to make this an EGL extension.
        
    2.  Should the API support multipassing?
    
        Decision - No.
        
        Reason - Multipassing should really be left to the application to do.  
        This makes the API unnecessarily complicated.  A major issue is that 
        depending on which counters are to be sampled, the # of passes and which 
        counters get selected in each pass can be difficult to determine.  It is 
        much easier to give a list of counters categorized by groups with 
        specific information on the number of counters that can be selected from 
        each group.

    3.  Should we define a 64-bit data type for UNSIGNED_INT64_AMD?

        Decision - No.

        Reason - While counters can be returned as 64-bit unsigned integers, the
        data is passed back to the application inside of a void*.  Therefore,
        there is no need in this extension to define a 64-bit data type (e.g.,
        GLuint64).  It will be up the application to declare a native 64-bit
        unsigned integer and cast the returned data to that type.


New Procedures and Functions

    void GetPerfMonitorGroupsAMD(int *numGroups, sizei groupsSize, 
                                 uint *groups)
    
    void GetPerfMonitorCountersAMD(uint group, int *numCounters, 
                                   int *maxActiveCounters, sizei countersSize, 
                                   uint *counters)

    void GetPerfMonitorGroupStringAMD(uint group, sizei bufSize, sizei *length, 
                                      char *groupString)

    void GetPerfMonitorCounterStringAMD(uint group, uint counter, sizei bufSize,
                                        sizei *length, char *counterString)
 
    void GetPerfMonitorCounterInfoAMD(uint group, uint counter, 
                                      enum pname, void *data)
    
    void GenPerfMonitorsAMD(sizei n, uint *monitors)
    
    void DeletePerfMonitorsAMD(sizei n, uint *monitors)
    
    void SelectPerfMonitorCountersAMD(uint monitor, boolean enable, 
                                      uint group, int numCounters, 
                                      uint *counterList)

    void BeginPerfMonitorAMD(uint monitor)
        
    void EndPerfMonitorAMD(uint monitor)

    void GetPerfMonitorCounterDataAMD(uint monitor, enum pname, sizei dataSize, 
                                      uint *data, int *bytesWritten)


New Tokens

    Accepted by the <pame> parameter of GetPerfMonitorCounterInfoAMD
    
        COUNTER_TYPE_AMD                           0x8BC0
        COUNTER_RANGE_AMD                          0x8BC1
        
    Returned as a valid value in <data> parameter of
    GetPerfMonitorCounterInfoAMD if <pname> = COUNTER_TYPE_AMD
        
        UNSIGNED_INT                               0x1405
        FLOAT                                      0x1406
        UNSIGNED_INT64_AMD                         0x8BC2
        PERCENTAGE_AMD                             0x8BC3
        
    Accepted by the <pname> parameter of GetPerfMonitorCounterDataAMD
        
        PERFMON_RESULT_AVAILABLE_AMD               0x8BC4
        PERFMON_RESULT_SIZE_AMD                    0x8BC5
        PERFMON_RESULT_AMD                         0x8BC6

Addition to the GL specification

    Add a new section called Performance Monitoring
    
    A performance monitor consists of a number of hardware and software counters
    that can be sampled by the GPU and reported back to the application.
    Performance counters are organized as a single hierarchy where counters are
    categorized into groups.  Each group has a list of counters that belong to
    the counter and can be sampled, and a maximum number of counters that can be 
    sampled.
    
    The command
    
        void GetPerfMonitorGroupsAMD(int *numGroups, sizei groupsSize, 
                                     uint *groups);
        
    returns the number of available groups in <numGroups>, if <numGroups> is
    not NULL.  If <groupsSize> is not 0 and <groups> is not NULL, then the list 
    of available groups is returned.  The number of entries that will be 
    returned in <groups> is determined by <groupsSize>.  If <groupsSize> is 0, 
    no information is copied.  Each group is identified by a unique unsigned int 
    identifier.
    
    The command
    
        void GetPerfMonitorCountersAMD(uint group, int *numCounters, 
                                       int *maxActiveCounters, 
                                       sizei countersSize, 
                                       uint *counters);
        
    returns the following information.  For each group, it returns the number of 
    available counters in <numCounters>, the max number of counters that can be
    active at any time in <maxActiveCounters>, and the list of counters in 
    <counters>.  The number of entries that can be returned in <counters> is
    determined by <countersSize>.  If <countersSize> is 0, no information is
    copied. Each counter in a group is identified by a unique unsigned int
    identifier.  If <group> does not reference a valid group ID, an 
    INVALID_VALUE error is generated.

    
    The command
    
        void GetPerfMonitorGroupStringAMD(uint group, sizei bufSize, 
                                          sizei *length, char *groupString)

        
    returns the string that describes the group name identified by <group> in 
    <groupString>.  The actual number of characters written to <groupString>,
    excluding the null terminator, is returned in <length>.  If <length> is 
    NULL, then no length is returned.  The maximum number of characters that
    may be written into <groupString>, including the null terminator, is 
    specified by <bufSize>.  If <bufSize> is 0 and <groupString> is NULL, the 
    number of characters that would be required to hold the group string,
    excluding the null terminator, is returned in <length>.  If <group> 
    does not reference a valid group ID, an INVALID_VALUE error is generated.
    
    
    The command
    
        void GetPerfMonitorCounterStringAMD(uint group, uint counter, 
                                            sizei bufSize, sizei *length, 
                                            char *counterString);

    
    returns the string that describes the counter name identified by <group> 
    and <counter> in <counterString>.  The actual number of characters written 
    to <counterString>, excluding the null terminator, is returned in <length>.  
    If <length> is NULL, then no length is returned.  The maximum number of 
    characters that may be written into <counterString>, including the null 
    terminator, is specified by <bufSize>.  If <bufSize> is 0 and 
    <counterString> is NULL, the number of characters that would be required to 
    hold the counter string, excluding the null terminator, is returned in 
    <length>.  If <group> does not reference a valid group ID, or <counter> 
    does not reference a valid counter within the group ID, an INVALID_VALUE 
    error is generated.
       
    The command
    
        void GetPerfMonitorCounterInfoAMD(uint group, uint counter, 
                                          enum pname, void *data);
        
    returns the following information about a counter.  For a <counter> 
    belonging to <group>, we can query the counter type and counter range.  If 
    <pname> is COUNTER_TYPE_AMD, then <data> returns the type.  Valid type
    values returned are UNSIGNED_INT, UNSIGNED_INT64_AMD, PERCENTAGE_AMD, FLOAT.
    If type value returned is PERCENTAGE_AMD, then this describes a float
    value that is in the range [0.0 .. 100.0].  If <pname> is COUNTER_RANGE_AMD,
    <data> returns two values representing a minimum and a maximum. The 
    counter's type is used to determine the format in which the range values 
    are returned.  If <group> does not reference a valid group ID, or <counter> 
    does not reference a valid counter within the group ID, an INVALID_VALUE 
    error is generated.

    
    The command
    
        void GenPerfMonitorsAMD(sizei n, uint *monitors)
        
    returns a list of monitors.  These monitors can then be used to select 
    groups/counters to be sampled, to start multiple monitoring sessions and to 
    return counter information sampled by the GPU.  At creation time, the 
    performance monitor object has all counters disabled.  The value of the
    PERFMON_RESULT_AVAILABLE_AMD, PERFMON_RESULT_AMD, and 
    PERFMON_RESULT_SIZE_AMD queries will all initially be 0.
    
    The command
    
        void DeletePerfMonitorsAMD(sizei n, uint *monitors)
        
    is used to delete the list of monitors created by a previous call to 
    GenPerfMonitors.  If a monitor ID in the list <monitors> does not 
    reference a previously generated performance monitor, an INVALID_VALUE
    error is generated.
    
    The command 
    
        void SelectPerfMonitorCountersAMD(uint monitor, boolean enable, 
                                          uint group, int numCounters, 
                                          uint *counterList);
        
    is used to enable or disable a list of counters from a group to be monitored 
    as identified by <monitor>.  The <enable> argument determines whether the
    counters should be enabled or disabled.  <group> specifies the group
    ID under which counters will be enabled or disabled.  The <numCounters>
    argument gives the number of counters to be selected from the list 
    <counterList>.  If <monitor> is not a valid monitor created by 
    GenPerfMonitorsAMD, then INVALID_VALUE error will be generated.  If <group>
    is not a valid group, the INVALID_VALUE error will be generated.  If 
    <numCounters> is less than 0, an INVALID_VALUE error will be generated. 

    When SelectPerfMonitorCountersAMD is called on a monitor, any outstanding 
    results for that monitor become invalidated and the result queries 
    PERFMON_RESULT_SIZE_AMD and PERFMON_RESULT_AVAILABLE_AMD are reset to 0.
    
    The command
    
        void BeginPerfMonitorAMD(uint monitor);
        
    is used to start a monitor session.  Note that BeginPerfMonitor calls cannot 
    be nested.  In addition, it is quite possible that given the list of groups 
    and counters/group enabled for a monitor, it may not be able to sample the 
    necessary counters and so the monitor session will fail.  In such a case,
    an INVALID_OPERATION error will be generated.

    While BeginPerfMonitorAMD does mark the beginning of performance counter
    collection, the counters do not begin collecting immediately.  Rather, the
    counters begin collection when BeginPerfMonitorAMD is processed by
    the hardware.  That is, the API is asynchronous, and performance counter
    collection does not begin until the graphics hardware processes the
    BeginPerfMonitorAMD command.  
    
    The command
    
        void EndPerfMonitorAMD(uint monitor);
        
    ends a monitor session started by BeginPerfMonitorAMD.  If a performance 
    monitor is not currently started, an INVALID_OPERATION error will be 
    generated.
    
    Note that there is an implied overhead to collecting performance counters
    that may or may not distort performance depending on the implementation.  
    For example, some counters may require a pipeline flush thereby causing a
    change in the performance of the application.  Further, the frequency at 
    which an application samples may distort the accuracy of counters which are 
    variant (e.g., non-deterministic based on the input).  While the effects 
    of sampling frequency are implementation dependent, general guidance can
    be given that sampling at a high frequency may distort both performance
    of the application and the accuracy of variant counters.

    The command
    
        void GetPerfMonitorCounterDataAMD(uint monitor, enum pname, 
                                          sizei dataSize, 
                                          uint *data, sizei *bytesWritten);
        
    is used to return counter values that have been sampled for a monitor
    session.  If <pname> is PERFMON_RESULT_AVAILABLE_AMD, then <data> will
    indicate whether the result is available or not.  If <pname> is 
    PERFMON_RESULT_SIZE_AMD, <data> will contain actual size of all counter 
    results being sampled.  If <pname> is PERFMON_RESULT_AMD, <data> will
    contain results.  For each counter of a group that was selected to be 
    sampled, the information is returned as group ID, followed by counter ID, 
    followed by counter value.  The size of counter value returned will depend 
    on the counter value type.  The argument <dataSize> specifies the number of
    bytes available in the <data> buffer for writing.  If <bytesWritten> is not 
    NULL, it gives the number of bytes written into the <data> buffer.  It is an 
    INVALID_OPERATION error for <data> to be NULL.  If <pname> is 
    PERFMON_RESULT_AMD and <dataSize> is less than the number of bytes required 
    to store the results as reported by a PERFMON_RESULT_SIZE_AMD query, then 
    results will be written only up to the number of bytes specified by 
    <dataSize>.

    If no BeginPerfMonitorAMD/EndPerfMonitorAMD has been issued for a monitor,
    then the result of querying for PERFMON_RESULT_AVAILABLE and 
    PERFMON_RESULT_SIZE will be 0.  When SelectPerfMonitorCountersAMD is called
    on a monitor, the results stored for the monitor become invalidated and
    the value of PERFMON_RESULT_AVAILABLE and PERFMON_RESULT_SIZE queries should
    behave as if no BeginPerfMonitorAMD/EndPerfMonitorAMD has been issued for
    the monitor.

Errors

    INVALID_OPERATION error will be generated if BeginPerfMonitorAMD is unable
    to begin monitoring with the currently selected counters.  

    INVALID_OPERATION error will be generated if BeginPerfMonitorAMD is called
    when a performance monitor is already active.

    INVALID_OPERATION error will be generated if EndPerfMonitorAMD is called
    when a performance monitor is not currently started.

    INVALID_VALUE error will be generated if the <group> parameter to 
    GetPerfMonitorCountersAMD, GetPerfMonitorCounterStringAMD,
    GetPerfMonitorCounterStringAMD, GetPerfMonitorCounterInfoAMD, or
    SelectPerfMonitorCountersAMD does not reference a valid group ID.

    INVALID_VALUE error will be generated if the <counter> parameter to
    GetPerfMonitorCounterInfoAMD does not reference a valid counter ID
    in the group specified by <group>.

    INVALID_VALUE error will be generated if any of the monitor IDs
    in the <monitors> parameter to DeletePerfMonitorsAMD do not reference
    a valid generated monitor ID.
   
    INVALID_VALUE error will be generated if the <monitor> parameter to
    SelectPerfMonitorCountersAMD does not reference a monitor created by
    GenPerfMonitorsAMD.

    INVALID_VALUE error will be generated if the <numCounters> parameter to
    SelectPerfMonitorCountersAMD is less than 0.

     

New State

Sample Usage

    typedef struct 
    {
            GLuint       *counterList;
            int         numCounters;
            int         maxActiveCounters;
    } CounterInfo;

    void
    getGroupAndCounterList(GLuint **groupsList, int *numGroups, 
                           CounterInfo **counterInfo)
    {
        GLint          n;
        GLuint        *groups;
        CounterInfo   *counters;

        glGetPerfMonitorGroupsAMD(&n, 0, NULL);
        groups = (GLuint*) malloc(n * sizeof(GLuint));
        glGetPerfMonitorGroupsAMD(NULL, n, groups);
        *numGroups = n;

        *groupsList = groups;
        counters = (CounterInfo*) malloc(sizeof(CounterInfo) * n);
        for (int i = 0 ; i < n; i++ )
        {
            glGetPerfMonitorCountersAMD(groups[i], &counters[i].numCounters,
                                     &counters[i].maxActiveCounters, 0, NULL);

            counters[i].counterList = (GLuint*)malloc(counters[i].numCounters * 
                                                      sizeof(int));

            glGetPerfMonitorCountersAMD(groups[i], NULL, NULL,
                                        counters[i].numCounters, 
                                        counters[i].counterList);
        }

        *counterInfo = counters;
    }
    
    static int  countersInitialized = 0;
        
    int
    getCounterByName(char *groupName, char *counterName, GLuint *groupID, 
                     GLuint *counterID)
    {
        int          numGroups;
        GLuint       *groups;
        CounterInfo  *counters;
        int          i = 0;

        if (!countersInitialized)
        {
            getGroupAndCounterList(&groups, &numGroups, &counters);
            countersInitialized = 1;
        }

        for ( i = 0; i < numGroups; i++ )
        {
           char curGroupName[256];
           glGetPerfMonitorGroupStringAMD(groups[i], 256, NULL, curGroupName);
           if (strcmp(groupName, curGroupName) == 0)
           {
               *groupID = groups[i];
               break;
           }
        }

        if ( i == numGroups )
            return -1;           // error - could not find the group name

        for ( int j = 0; j < counters[i].numCounters; j++ )
        {
            char curCounterName[256];
            
            glGetPerfMonitorCounterStringAMD(groups[i],
                                             counters[i].counterList[j], 
                                             256, NULL, curCounterName);
            if (strcmp(counterName, curCounterName) == 0)
            {
                *counterID = counters[i].counterList[j];
                return 0;
            }
        }

        return -1;           // error - could not find the counter name
    }

    void
    drawFrameWithCounters(void)
    {
        GLuint group[2];
        GLuint counter[2];
        GLuint monitor;
        GLuint *counterData;

        // Get group/counter IDs by name.  Note that normally the
        // counter and group names need to be queried for because
        // each implementation of this extension on different hardware
        // could define different names and groups.  This is just provided
        // to demonstrate the API.
        getCounterByName("HW", "Hardware Busy", &group[0],
                         &counter[0]);
        getCounterByName("API", "Draw Calls", &group[1], 
                         &counter[1]);
                
        // create perf monitor ID
        glGenPerfMonitorsAMD(1, &monitor);

        // enable the counters
        glSelectPerfMonitorCountersAMD(monitor, GL_TRUE, group[0], 1,
                                       &counter[0]);
        glSelectPerfMonitorCountersAMD(monitor, GL_TRUE, group[1], 1, 
                                       &counter[1]);

        glBeginPerfMonitorAMD(monitor);

        // RENDER FRAME HERE
        // ...
        
        glEndPerfMonitorAMD(monitor);

        // read the counters
        GLint resultSize;
        glGetPerfMonitorCounterDataAMD(monitor, GL_PERFMON_RESULT_SIZE_AMD, 
                                       sizeof(GLint), &resultSize, NULL);

        counterData = (GLuint*) malloc(resultSize);

        GLsizei bytesWritten;
        glGetPerfMonitorCounterDataAMD(monitor, GL_PERFMON_RESULT_AMD,  
                                       resultSize, counterData, &bytesWritten);

        // display or log counter info
        GLsizei wordCount = 0;

        while ( (4 * wordCount) < bytesWritten )
        {
            GLuint groupId = counterData[wordCount];
            GLuint counterId = counterData[wordCount + 1];

            // Determine the counter type
            GLuint counterType;
            glGetPerfMonitorCounterInfoAMD(groupId, counterId, 
                                           GL_COUNTER_TYPE_AMD, &counterType);
 
            if ( counterType == GL_UNSIGNED_INT64_AMD )
            {
                unsigned __int64 counterResult = 
                           *(unsigned __int64*)(&counterData[wordCount + 2]);

                // Print counter result

                wordCount += 4;
            }
            else if ( counterType == GL_FLOAT )
            {
                float counterResult = *(float*)(&counterData[wordCount + 2]);

                // Print counter result

                wordCount += 3;
            } 
            // else if ( ... ) check for other counter types 
            //   (GL_UNSIGNED_INT and GL_PERCENTAGE_AMD)
        }
    }
 
Revision History
    11/29/2007 - dginsburg
       + Clarified the default state of a performance monitor object on creation

    11/09/2007 - dginsbur
       + Clarify what happens if SelectPerfMonitorCountersAMD is called on
         a monitor with outstanding query results.
       + Rename counterSize to countersSize
       + Remove some ';' typos

    06/13/2007 - dginsbur
       + Add language on the asynchronous nature of the API and 
         counter accuracy/performance distortion.
       + Add myself as the contact
       + Remove INVALID_OPERATION error when countersList is NULL
       + Clarify 64-bit issue
       + Make PERCENTAGE_AMD counters float rather than uint
       + Clarify accuracy distortion on variant counters only
       + Tweak to overview language

    06/09/2007 - dginsbur
       + Fill in errors section and make many more errors explicit
       + Fix the example code so it compiles

    06/08/2007 - dginsbur
       + Modified GetPerfMonitorGroupString and GetPerfMonitorCounterString to
         be more client/server friendly.  
       + Modified example.
       + Renamed parameters/variables to follow GL conventions.
       + Modified several 'int' param types to 'sizei'
       + Modifid counters type from 'int' to 'uint'
       + Renamed argument 'cb' and 'cbret'
       + Better documented GetPerfMonitorCounterData 
       + Add AMD adornment in many places that were missing it
 
    06/07/2007 - dginsbur
       + Cleanup formatting, remove tabs, make fit in proper page width
       + Add FLOAT and UNSIGNED_INT to list of COUNTER_TYPEs
       + Fix some bugs in the example code
       + Rewrite introduction
       + Clarified Issue 1 reasoning
       + Added Issue 3 regarding use of 64-bit data types
       + Added revision history

    03/21/2007 - Initial version written.  Written by amunshi.

        
