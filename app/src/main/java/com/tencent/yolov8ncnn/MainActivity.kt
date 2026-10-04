package com.tencent.yolov8ncnn

import android.Manifest
import android.app.Activity
import android.content.pm.PackageManager
import android.graphics.PixelFormat
import android.os.Bundle
import android.util.Log
import android.view.SurfaceHolder
import android.view.SurfaceView
import android.view.View
import android.view.WindowManager
import android.widget.AdapterView
import android.widget.Button
import android.widget.Spinner

import android.support.v4.app.ActivityCompat
import android.support.v4.content.ContextCompat

class MainActivity : Activity(), SurfaceHolder.Callback{
    private val yolov8ncnn = Yolov8Ncnn()
    private var facing = 0
    private lateinit var spinnerModel: Spinner
    private lateinit var spinnerCPUGPU: Spinner
    private var currentModel = 0
    private var currentCpuGpu = 0

    private lateinit var  cameraView: SurfaceView

    override fun onCreate(savedInstanceState:Bundle?){
        super.onCreate(savedInstanceState)
        setContentView(R.layout.main)

        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        cameraView = findViewById(R.id.cameraview)
        cameraView.holder.setFormat(PixelFormat.RGBA_8888)
        cameraView.holder.addCallback(this)

        val buttonSwitchCamera = findViewById<Button>(R.id.buttonSwitchCamera)
        buttonSwitchCamera.setOnClickListener{
            val newFacing = 1 - facing
            yolov8ncnn.closeCamera()
            yolov8ncnn.openCamera(newFacing)
            facing = newFacing
        }

        spinnerModel = findViewById(R.id.spinnerModel)
        spinnerModel.onItemSelectedListener = object : AdapterView.OnItemSelectedListener{

            override fun onItemSelected(parent : AdapterView<*>?, View: View?, position: Int, id: Long){
                if(position != currentModel){
                    currentModel = position
                    reload()

                }
            }
            override fun onNothingSelected(parent: AdapterView<*>?){}
        }
        spinnerCPUGPU = findViewById(R.id.spinnerCPUGPU)
        spinnerCPUGPU.onItemSelectedListener = object : AdapterView.OnItemSelectedListener{
            override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                if(position != currentCpuGpu){
                    currentCpuGpu = position
                    reload()
                }
            }

            override fun onNothingSelected(parent: AdapterView<*>?) {}
        }
        reload()

    }
    private fun reload(){
        val reInit = yolov8ncnn.loadModel(assets,currentModel, currentCpuGpu)
        if(!reInit){
            Log.e("MainActivity","yolov8ncnn loadModel failed")
        }
    }
    override fun surfaceChanged(holder: SurfaceHolder, format:Int, width: Int, height:Int){
        yolov8ncnn.setOutputWindow(holder.surface)
    }

    override fun surfaceCreated(holder: SurfaceHolder) {}
    override fun surfaceDestroyed(holder: SurfaceHolder){}

    override fun onResume(){
        super.onResume()

        val denied = ContextCompat.checkSelfPermission(
            applicationContext,
            Manifest.permission.CAMERA
        ) == PackageManager.PERMISSION_DENIED
        if (denied){
            ActivityCompat.requestPermissions(this,arrayOf(Manifest.permission.CAMERA), REQUEST_CAMERA)
        }else{
            yolov8ncnn.openCamera(facing)
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)

        if(requestCode == REQUEST_CAMERA){
            if(grantResults.isNotEmpty()&&grantResults[0] == PackageManager.PERMISSION_GRANTED){
                yolov8ncnn.openCamera(facing)
            }
        }
    }
    override fun onPause(){
        super.onPause()
        yolov8ncnn.closeCamera()
    }
    companion object{
        const val REQUEST_CAMERA = 100
    }
}