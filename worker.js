/**
	* 
	* Example usage in main thread.
	*
	* 	const {default: Worker} = await import(
	*   	'@spriteful/mobilenet-worker/worker.js'
	* 	);
	*
	* Use new Comlink v4+ library
	*
	* 	const worker = Comlink.wrap(new Worker());
	*
	* Optionally set a custom path that points to the 
	* model directory.
	* Don't forget to copy the folder over in webpack.
	*
	* 	await worker.load('./my-model-dir');
	*
	* Use new Comlink v4+ transfer method
	* to transfer instead of copy the ImageBitmap.
	* This equates to a big performance gain.
	*
	* 	const prediction = await worker.predict(
	*   	Comlink.transfer({bitmap}, [bitmap]) 
	* 	);
	*
	*
	**/

// match mobilenet input
const IMAGE_SIZE = 224;
// layers model
const MOBILENET_URL = 'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_1.0_224/model.json';
// tf in a web worker polyfill
// MUST be before importing tf, as of v1.2.2
// see - https://github.com/tensorflow/tfjs/issues/102
self.document = {
  createElement: () => {
    return new OffscreenCanvas(IMAGE_SIZE, IMAGE_SIZE);
  }
};
self.window = self;
self.screen = {
  width: 640,
  height: 480
};
self.HTMLVideoElement  = function() {};
self.HTMLImageElement  = function() {};
self.HTMLCanvasElement = OffscreenCanvas;


import * as Comlink from 'comlink';
import * as tf 		  from '@tensorflow/tfjs';


let classLabels;
let netModel;
let retrainedModel;
// this canvas gets reused to crop incoming bitmap to fit mobilenet
const canvas  = new OffscreenCanvas(IMAGE_SIZE, IMAGE_SIZE);
const context = canvas.getContext('2d');

// load mobilenet and keep all but top layers
const loadDecapitatedMobilenet = async url => {
  const mobilenet = await tf.loadLayersModel(url);
  // Return a model that outputs an internal activation.
  const layer = mobilenet.getLayer('conv_pw_13_relu');
  return tf.model({
  	inputs:  mobilenet.inputs, 
  	outputs: layer.output
  });
};

// load mobilenet base model and custom retrained layers model
const load = async (modelDir = './model') => {
	const modelPath  = `${modelDir}/model.json`;
	const labelsPath = `${modelDir}/labels.json`;
	const response 	 = await fetch(labelsPath);
	const labels 		 = await response.json();
	classLabels 		 = labels['Labels'];
	netModel 			 	 = await loadDecapitatedMobilenet(MOBILENET_URL);
	retrainedModel 	 = await tf.loadLayersModel(modelPath);
	// console.log('backend: ', tf.getBackend());
};

// center and crop to square
const cropImageToSquare = bitmap => {	
  const scale  		= IMAGE_SIZE / Math.min(bitmap.height, bitmap.width);
  const landscape = bitmap.width > bitmap.height;
  const width  		= scale * bitmap.width;
  const height 		= scale * bitmap.height;
  const x = landscape ? (width - IMAGE_SIZE) / -2 : 0;  
  const y = landscape ? 0 : (height - IMAGE_SIZE) / -2;
  context.drawImage(bitmap, x, y, width, height);
  bitmap.close(); // memory management
  return canvas;
};


const imgDataToTensor = img => {
	return tf.tidy(() => {	// tf memory management
		const data 	 = tf.browser.fromPixels(img);
		// Normalize the image from [0, 255] to [-1, 1].
		// must match that of 'node-tfjs-retrain' module
		const normalized = data.div(tf.scalar(127)).sub(tf.scalar(1));
	  // Reshape to a single-element batch so we can pass it to predict.
	  const batched = normalized.reshape([1, IMAGE_SIZE, IMAGE_SIZE, 3]);
	  // log tensor info
	  // batched.print(true);
	  return batched;
	});
};

// // memory management (use tf.memory() to debug)
// const reset = () => {
// 	 tf.disposeVariables();
// };


const predict = async ({bitmap}) => {
	const img 			 = cropImageToSquare(bitmap);
	const input  		 = imgDataToTensor(img);
	const embeddings = await netModel.predict(input);
	const output 		 = await retrainedModel.predict(embeddings);
	// top 5 classes, sorted highest first
	const {indices, values} = output.topk(5, true);
	const [
		indicesData, 
		valuesData
	] = await Promise.all([
		indices.data(), 
		values.data()
	]); 
	// serve a collection of confidence levels and corresponding labels
	const predictions = Array.from(valuesData).map((confidence, index) => {
		const label = classLabels[indicesData[index]];
		return {confidence, label};
	});

	// memory management (use tf.memory() to debug)
	input.dispose();
	embeddings.dispose();
	output.dispose();
	indices.dispose();
	values.dispose();
	await tf.nextFrame();
	return predictions;
};


const api = {
	load,
	predict
};


Comlink.expose(api, self);
