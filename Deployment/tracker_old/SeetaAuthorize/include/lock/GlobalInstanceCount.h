#ifndef _GLOBALINSTANCECOUNT_H_
#define _GLOBALINSTANCECOUNT_H_

#include <string>

#ifdef WIN32
    #include <windows.h>
#endif


class GlobalInstanceCount {
public:
    //name:"seetatech.com" + "." + appname, the length <= 24, example name:"seetatech.com.authorize"
    GlobalInstanceCount( const std::string &name );
    ~GlobalInstanceCount();

    //return -1:failed, 0:ok
    int Open();
    //return > 0, ok, 0:failed
    int InstanceCount();
private:
    std::string m_name;


    #ifdef WIN32
    HANDLE m_handle;
    #else
    int m_shmid;
    void *m_local;
    #endif
};

#endif
