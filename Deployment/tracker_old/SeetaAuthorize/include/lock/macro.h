#pragma once

// means SDK has right to call func_id, could be number or string with quotation
// #define SEETA_LOCK_ABILITY 1001 1002

// if set, mean using online lock checking SEETA_LOCK_FINCID
// #define SEETA_LOCK_SDK_ONLINE

// if set, using SEETA_LOCK_ABILITY checking SEETA_LOCK_FINCID, So do model checking
// #define SEETA_LOCK_SDK_LOCAL

// means SDK API need check funcid, should be number
// #define SEETA_LOCK_FINCID 1001

// if set, using lock model
// #define SEETA_LOCK_MODEL

// means SDK user key, could be string with or with out quotation
// #define SEETA_LOCK_KEY XJ37DUTJ

// means SDK model lock key, could be string with or with out quotation
// #define SEETA_LOCK_MODEL_KEY XJ37DUTJ

// notice if SEETA_LOCK_SDK_ONLINE, SEETA_LOCK_SDK_LOCAL or SEETA_LOCK_MODEL is set, SEETA_LOCK_SDK should be set

#if defined(SEETA_LOCK_SDK_ONLINE) && !defined(SEETA_LOCK_KEY)
    #error Must define SEETA_LOCK_KEY when using SEETA_LOCK_SDK_ONLINE
#endif

#if defined(SEETA_LOCK_SDK_LOCAL) && !defined(SEETA_LOCK_KEY)
    #error Must define SEETA_LOCK_KEY when using SEETA_LOCK_SDK_LOCAL
#endif

#if defined(SEETA_LOCK_SDK_LOCAL) && !defined(SEETA_LOCK_ABILITY)
    #error Must define SEETA_LOCK_ABILITY when using SEETA_LOCK_SDK_LOCAL
#endif

#if defined(SEETA_LOCK_MODEL) && !defined(SEETA_LOCK_KEY) && !defined(SEETA_LOCK_MODEL_KEY)
    #error Must define SEETA_LOCK_KEY or SEETA_LOCK_MODEL_KEY when using SEETA_LOCK_MODEL
#endif

// lockit should lower here
#include "lockit.h"

#if defined(SEETA_LOCK_SDK_ONLINE)
    // check online, controled by SEETA_LOCK_SDK_ONLINE
    #define SEETA_CHECK_INIT seeta_check_init();
#endif

#if defined(SEETA_LOCK_SDK_ONLINE)
    // check online, controled by SEETA_LOCK_SDK_ONLINE
    #define SEETA_CHECK_FUNCID(funcid, msg) seeta_check_funcid((funcid), (msg));
#endif

#if defined(SEETA_LOCK_SDK_ONLINE)
    // check online, controled by SEETA_LOCK_SDK_ONLINE
    #define SEETA_CHECK_ALL_FUNCID(funcid, msg) seeta_check_all_funcid((funcid), (msg));
#endif

#if defined(SEETA_LOCK_SDK_ONLINE)
    // check online, controled by SEETA_LOCK_SDK_ONLINE
    #define SEETA_CHECK_ANY_FUNCID(funcid, msg) seeta_check_any_funcid((funcid), (msg));
#endif

#if defined(SEETA_LOCK_SDK_ONLINE) && (SEETA_LOCK_FINCID)
    // check online, controled by SEETA_LOCK_SDK_ONLINE
    #define SEETA_CHECK_AUTO_FUNCID(msg) seeta_check_auto_funcid((msg));
#endif

#if defined(SEETA_LOCK_SDK_ONLINE) || defined(SEETA_LOCK_SDK_LOCAL)
    // check online or local, controled by SEETA_LOCK_SDK_LOCAL or SEETA_LOCK_SDK_ONLINE
    #define SEETA_CHECK_MODEL(model, msg) seeta_check_model((model), (msg));
#endif

#if defined(SEETA_LOCK_MODEL)
    // check model, controled by SEETA_LOCK_MODEL
    #define SEETA_CHECK_LOAD(bin) seeta_check_load(bin)
#endif

#if defined(TIME_LOCK)
    // check time, controled by TIME_LOCK
    #define SEETA_CHECK_TIME seeta_check_time();
#endif

