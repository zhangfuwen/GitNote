#include <iostream>
int read_cpu()
{
    char cpu_model[128] = "";
    FILE *fp = fopen("/proc/cpuinfo", "r");
    if (fp != NULL)
    {
        char name[32] = "";
        while (fscanf(fp, "%s", name) > 0)
        {
            if (strcmp(name, "model") == 0 && strcmp(cpu_model, "") == 0)
            {
                fscanf(fp, "name : %[^\n]", cpu_model);
            }
            memset(name,0,sizeof name);
        }
        fclose(fp);
    }
    if (strcmp(cpu_model, "") == 0)
    {
        printf("failed to get cpu model\n");
    }
    else
    {
        printf("CPU Model: %s\n", cpu_model);
    }
    return 0;
}

int read_mem()
{
    char mem_model[128] = "";
    FILE *fp = fopen("/sys/devices/platform/soc/c047000.memory-controller/mem_type", "r");
    if (fp != NULL)
    {
        fscanf(fp, "%[^\n]", mem_model);
        fclose(fp);
    }
    if (strcmp(mem_model, "") == 0)
    {
        printf("failed to get memory model\n");
    }
    else
    {
        printf("Memory Model: %s\n", mem_model);
    }
    return 0;
}
int read_cpu1()
{
    char cpu_model[128] = "";
    FILE *fp = fopen("/proc/cpuinfo", "r");
    if (fp != NULL)
    {
        char name[32] = "";
        while (fscanf(fp, "%s", name) > 0)
        {
            if (strcmp(name, "model") == 0 && strcmp(cpu_model, "") == 0)
            {
                fscanf(fp, "name : %[^\n]", cpu_model);
            }
            memset(name,0,sizeof name);
        }
        fclose(fp);
    }
    if (strcmp(cpu_model, "") == 0)
    {
        printf("failed to get cpu model\n");
    }
    else
    {
        printf("CPU Model: %s\n", cpu_model);
    }
    return 0;
}

int read_mem1()
{
    char mem_model[128] = "";
    FILE *fp = fopen("/sys/devices/platform/soc/c047000.memory-controller/mem_type", "r");
    if (fp != NULL)
    {
        fscanf(fp, "%[^\n]", mem_model);
        fclose(fp);
    }
    if (strcmp(mem_model, "") == 0)
    {
        printf("failed to get memory model\n");
    }
    else
    {
        printf("Memory Model: %s\n", mem_model);
    }
    return 0;
}

int main() {
    std::cout << "hello" << std::endl;

    read_cpu();
    read_mem();
    return 0;
}
