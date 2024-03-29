# QCOM_performance_monitor_global_mode

Name

    QCOM_performance_monitor_global_mode

Name Strings

    GL_QCOM_perfmon_global_mode

Contributors

    Jari Komppa
    James Ritts
    Jeff Leger
    Maurice Ribble

Contact

    Jari Komppa, Qualcomm (jari.komppa 'at' qualcomm.com)

Status

    Complete.

Version

    Last Modified Date: May 5, 2009
    Revision: #03

Number

    OpenGL ES Extension #56

Dependencies

    This extension interacts with AMD_performance_monitor.

Overview

    This extension introduces a global tracking mode in the performance
    monitors. When enabled, the counters increment in all operations,
    including all "housekeeping" operations such as resolves/clears and
    operations performed by other concurrent applications. The normal
    operation is to track only the application's own operations.

Issues

    (1)  How do we filter operations from other applications?

    RESOLVED: We don't. The user of this extension should be aware of the
    consequences.

    (2)  What should happen if the global mode is enabled or disabled while
         the performance monitors are in use?

    RESOLVED: The results in this case are undefined.

New Tokens

    Accepted by the <cap> parameter of Enable and Disable, and
    IsEnabled, and by the <pname> parameter of GetBooleanv, GetIntegerv,
    and GetFloatv:

        PERFMON_GLOBAL_MODE_QCOM                     0x8FA0

New Procedures and Functions

    None.

Addition to the GL specification

    Add this to the Performance Monitoring section that was added by
    AMD_performance_monitor:

   "A global tracking mode can be enabled and disabled with the generic
    Enable and Disable commands using the symbolic constant
    PERFMON_GLOBAL_MODE_QCOM. When enabled the counters increment in all
    operations, contexts, and other applications using the hardware."

Errors

    None

Sample Usage

    // Activate global performance monitor tracking
    glEnable(GL_PERFMON_GLOBAL_MODE_QCOM);

Revision History

    #01    18/02/2009    Jari Komppa     First draft.
    #02    19/03/2009    Maurice Ribble  Add in missing sections.
    #03    05/05/2009    Jon Leech       Reflow paragraphs and assign
                                         extension number.
