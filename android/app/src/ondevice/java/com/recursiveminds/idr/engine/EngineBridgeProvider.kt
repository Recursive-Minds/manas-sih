package com.recursiveminds.idr.engine

import android.content.Context

object EngineBridgeProvider {
    fun create(context: Context): IEngineBridge {
        return LocalChaquopyEngineBridge(context)
    }
}
