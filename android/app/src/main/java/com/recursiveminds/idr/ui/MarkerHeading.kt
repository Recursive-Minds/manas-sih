package com.recursiveminds.idr.ui

/**
 * T1 - vehicle pointer turns the wrong way.
 *
 * Every heading we produce (Android Location.bearing, server dr heading_deg,
 * EKF _heading_rad) is a COMPASS bearing: 0 = north, clockwise positive.
 * osmdroid's Marker.rotation is applied counter-clockwise on screen
 * (Marker.draw uses canvas.rotate(-rotation)), so passing a bearing directly
 * mirrors the pointer: a right turn rotates it left.
 *
 * Fix: convert once, here. If a device test ever shows the opposite, flip
 * OSMDROID_ROTATION_IS_CCW - nothing else needs to change.
 */
object MarkerHeading {
    const val OSMDROID_ROTATION_IS_CCW = true

    fun toMarkerRotation(bearingDeg: Double?): Float {
        val b = bearingDeg ?: 0.0
        val r = if (OSMDROID_ROTATION_IS_CCW) 360.0 - b else b
        return (((r % 360.0) + 360.0) % 360.0).toFloat()
    }

    fun toMarkerRotation(bearingDeg: Float?): Float = toMarkerRotation(bearingDeg?.toDouble())
}
