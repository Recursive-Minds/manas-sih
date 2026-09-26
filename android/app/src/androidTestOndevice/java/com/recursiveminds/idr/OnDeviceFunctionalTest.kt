package com.recursiveminds.idr

import android.content.Context
import android.content.Intent
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.uiautomator.By
import androidx.test.uiautomator.UiDevice
import androidx.test.uiautomator.Until
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Full Functional Instrumented Test Suite for SIH IDR On-Device Mode (SM-F127G).
 * Automates UI interaction, control buttons, benchmark replay, and lifecycle states.
 */
@RunWith(AndroidJUnit4::class)
class OnDeviceFunctionalTest {

    private lateinit var device: UiDevice
    private val targetPackage = "com.recursiveminds.idr.ondevice"
    private val defaultTimeout = 5000L

    @Before
    fun setUp() {
        device = UiDevice.getInstance(InstrumentationRegistry.getInstrumentation())
        device.wakeUp()
        
        // Launch main activity
        val context = ApplicationProvider.getApplicationContext<Context>()
        val intent = context.packageManager.getLaunchIntentForPackage(targetPackage)?.apply {
            addFlags(Intent.FLAG_ACTIVITY_CLEAR_TASK or Intent.FLAG_ACTIVITY_NEW_TASK)
        }
        context.startActivity(intent)
        device.wait(Until.hasObject(By.pkg(targetPackage).depth(0)), defaultTimeout)
    }

    @Test
    fun test01_launchAndHeader() {
        val header = device.wait(Until.findObject(By.text("SIH 26168")), defaultTimeout)
        assertNotNull("SIH 26168 header badge must be present", header)
        
        val modeText = device.findObject(By.text("Autonomous On-Device Engine"))
        assertNotNull("On-device engine label must be present", modeText)
    }

    @Test
    fun test02_buttonsPresence() {
        val btnStart = device.findObject(By.res(targetPackage, "btnStart"))
        val btnStop = device.findObject(By.res(targetPackage, "btnStop"))
        val btnReset = device.findObject(By.res(targetPackage, "btnReset"))
        val btnPrefetch = device.findObject(By.res(targetPackage, "btnPrefetchArea"))
        val btnToggleMap = device.findObject(By.res(targetPackage, "btnToggleMapMatching"))
        val btnBenchmark = device.findObject(By.res(targetPackage, "btnBenchmarkSuite"))
        val btnCsvMode = device.findObject(By.res(targetPackage, "btnCsvMode"))

        assertNotNull("START button must be present", btnStart)
        assertNotNull("STOP button must be present", btnStop)
        assertNotNull("RESET button must be present", btnReset)
        assertNotNull("PREFETCH button must be present", btnPrefetch)
        assertNotNull("MAP MATCH toggle must be present", btnToggleMap)
        assertNotNull("BENCHMARK button must be present", btnBenchmark)
        assertNotNull("CSV REC button must be present", btnCsvMode)
    }

    @Test
    fun test03_warningDialogBeforeYawLock() {
        val btnStart = device.wait(Until.findObject(By.res(targetPackage, "btnStart")), defaultTimeout)
        btnStart.click()

        // Expect warning dialog
        val dialogTitle = device.wait(Until.findObject(By.text("Calibration Incomplete")), defaultTimeout)
        if (dialogTitle != null) {
            val btnWait = device.findObject(By.text("Wait for calibration"))
            val btnAnyway = device.findObject(By.text("Start anyway"))
            assertNotNull("Wait button must be present", btnWait)
            assertNotNull("Start anyway button must be present", btnAnyway)

            // Test Wait dismissal
            btnWait.click()
            val dismissed = device.wait(Until.gone(By.text("Calibration Incomplete")), defaultTimeout)
            assertTrue("Dialog should dismiss on Wait", dismissed)
        }
    }

    @Test
    fun test04_mapMatchingToggle() {
        val btnToggleMap = device.wait(Until.findObject(By.res(targetPackage, "btnToggleMapMatching")), defaultTimeout)
        val initialText = btnToggleMap.text
        btnToggleMap.click()
        Thread.sleep(500)
        val updatedText = btnToggleMap.text
        assertNotEquals("Map match button text should toggle", initialText, updatedText)
        // Toggle back
        btnToggleMap.click()
        Thread.sleep(500)
    }

    @Test
    fun test05_benchmarkDrawerOpenAndClose() {
        val btnBenchmark = device.wait(Until.findObject(By.res(targetPackage, "btnBenchmarkSuite")), defaultTimeout)
        btnBenchmark.click()

        val drawerTitle = device.wait(Until.findObject(By.res(targetPackage, "cardBenchmark")), defaultTimeout)
        assertNotNull("Benchmark card drawer should be visible", drawerTitle)

        val btnClose = device.findObject(By.res(targetPackage, "btnCloseBenchmark"))
        assertNotNull("Close benchmark drawer button must be present", btnClose)
        btnClose.click()

        val drawerClosed = device.wait(Until.gone(By.res(targetPackage, "cardBenchmark")), defaultTimeout)
        assertTrue("Benchmark drawer should close", drawerClosed)
    }

    @Test
    fun test06_csvRecordingToggle() {
        val btnCsvMode = device.wait(Until.findObject(By.res(targetPackage, "btnCsvMode")), defaultTimeout)
        btnCsvMode.click()

        val csvCard = device.wait(Until.findObject(By.res(targetPackage, "cardCsvRecording")), defaultTimeout)
        assertNotNull("CSV card should be visible", csvCard)

        // Close CSV card
        btnCsvMode.click()
    }
}
