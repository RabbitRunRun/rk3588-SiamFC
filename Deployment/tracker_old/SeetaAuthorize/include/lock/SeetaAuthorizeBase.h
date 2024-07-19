#pragma once
#include <string>
#include <vector>
#include <map>
#include <mutex>
#include "orz/sync/canyon.h"

#include "GlobalInstanceCount.h"

struct func_info_elem
{
    int func_id;
    int instance_count;
    float expire_timestamp;
};


class SeetaChecker {

public:
    SeetaChecker();
public:
    using self = SeetaChecker;

    ~SeetaChecker();

    void add_path( const std::string &path );
    void reset_path();

    /**
    * \brief
    * \param func_id
    * \return
    * serial.json for key_code
    * seetatech_receipts for device_code, not in file this version
    * seetatech_receipts_out2 for auth_info, not in file this version
    * supporting thread calling
    */
    void check( int func_id, const std::string &err_msg = "" );

    /**
    * \brief
    * \param func_ids
    * \return
    * serial.json for key_code
    * seetatech_receipts for device_code, not in file this version
    * seetatech_receipts_out2 for auth_info, not in file this version
    * supporting thread calling
    * \note success if it has one permission of func_ids
    */
    void check_any( const std::vector<int> &func_ids, const std::string &err_msg = "" );

    /**
    * \brief
    * \param func_ids
    * \return
    * serial.json for key_code
    * seetatech_receipts for device_code, not in file this version
    * seetatech_receipts_out2 for auth_info, not in file this version
    * supporting thread calling
    * \note success if it has all permission of func_ids
    */
    void check_all( const std::vector<int> &func_ids, const std::string &err_msg = "" );

    /**
     * called in every object init, but only one init will really called
     * make sure after init, the object will be ready to use
         * return 0: success, -1: failed
     */
    void init();

    /**
     * get key_code
     */
    std::string get_key_code() const;

    std::string get_online_url() const;
    std::string get_version() const;

    int get_sdk_instances() const {
        return sdk_instances;
    }


    void check_global_instances(const char * name);

   
    /**
    * \read config file to init
    * \param
    * \return 0:success, -1:failed
    */
    void read_config();
private:
    /**
     * do not call this function this version
     */
    void update();


    void thread_check();
    /**
    * \read config file to init
    * \param
    * \return 0:success, -1:failed
    */
    //void read_config();

    std::string determine_types( const std::vector<std::string> &vec_types );
    std::string determine_type();
    void check_to_update();
    int dog_lock_update( std::map<int, func_info_elem> &key_functions, int &errorid, std::string &msg );
    int file_lock_update( int &errorid, std::string &msg );
    int online_update( std::map<int, func_info_elem> &key_functions, int &errorid );

    /**
    * \brief
     * \param func_id
     * \param reason [out] 如果为空，则没有异常，否则返回错误内容
     */
    void inner_check( int func_id, int &errorid );
    void raw_check( int func_id, int &errorid ) const;

    std::atomic<bool> valid;

    std::string key_code;
    std::string version;
    std::string device_code;
    /*
        struct func_info_elem
        {
            int func_id;
                    int instance_count;
            float expire_timestamp;
        };
    */
    struct auth_info_type
    {
        float timestamp;
        std::map<int, func_info_elem> finc_info_map;
    };

    mutable auth_info_type auth_info;

    mutable std::mutex update_mutex;
    //std::mutex init_mutex;
    //std::condition_variable init_cond;
    bool initialized = false;

    std::vector<std::string> receipts_path;

    std::vector<int> ability;

    std::map<int, func_info_elem> config_map;
    std::vector<std::string> types;

    std::string m_type;
    std::string online_url;

    std::string str_expire_time;
    time_t expire_time;
    int sdk_instances;

    std::vector<GlobalInstanceCount *> globalinstances;
};

