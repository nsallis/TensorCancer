const { train, math } = require('@tensorflow/tfjs-node-gpu');
const tf = require('@tensorflow/tfjs-node-gpu');
const fs = require('fs');
const parse = require('csv-parse');
const util = require('./util');
const _ = require('lodash');

const LEARNING_RATE = 0.05;
const EPOCHS = 100

const runTrain = async () => {
  const trainingData = fs.readFileSync('./data/data.csv');
  const data = await new Promise((resolve, reject) => {
    parse(trainingData, {relax_column_count: true}, (err, output) => {
      if (err) {
        reject(err);
      } else {
        resolve(output);
      }
    });
  });
  data.shift(); // remove headers

  const transformedData = _.shuffle(data).map(util.transformRow);

// Define the model.
const model = tf.sequential();
// Set up the network layers
model.add(tf.layers.dense({units: 5, activation: 'relu', inputShape: [10]}));
model.add(tf.layers.dense({units: 150, activation: 'relu'}));
model.add(tf.layers.dense({units: 250, activation: 'relu'}));
model.add(tf.layers.dense({units: 250, activation: 'relu'}));
model.add(tf.layers.dense({units: 250, activation: 'relu'}));
model.add(tf.layers.dense({units: 2, activation: 'softmax', outputShape: [2]}));
// Define the optimizer
const optimizer = tf.train.adam(LEARNING_RATE);
// Init the model
model.compile({
    optimizer: 'sgd',//optimizer,
    loss: 'binaryCrossentropy',
    metrics: ['accuracy'],
});

  const ys = transformedData.map(d => d.ys);
  const xs = transformedData.map(d => d.xs);

  let xTrain = tf.tensor2d(xs.slice(0,500), [xs.slice(0,500).length, 10]);
  let yTrain = tf.oneHot(tf.tensor1d(ys.slice(0,500)).toInt(), 2); // [yes, no] (yes malignant, no benign)

  console.log('ready to start training model');
  // await model.fit(xsTensor, tf.cast(ysTensor, 'int32'), {epochs: 1, batchSize: 100});
  const history = await model.fit(xTrain, yTrain, {
    epochs: EPOCHS,
    validationSplit: 0.3,
    shuffle: true
    // validationData: [xTrain, yTrain],
  })
  console.log('trained data');

  let mals = 0;
  let bens = 0;
  let success = 0;


  xs.slice(501).forEach((testXs, idx) => {
    const testYs = ys[idx];
    // const testVal = [0.206,	0.293,	0.140,	0.126,	0.117,	0.277,	0.3514,	0.152,	0.2397,	0.07016]
    console.log(`Testing row: ${JSON.stringify(testXs)} which should result in ${testYs ? 'Malignant' : 'Benign'} (${testYs})`);

    let result = model.predict(tf.tensor(testXs, [1, testXs.length])).arraySync();
    if (result[0][0] >= 0.5) {
      mals++;
    } else {
      bens++;
    }
    console.log(`Result: ${JSON.stringify(result)}`);
    console.log(`Test val evaluated as: ${result[0][0] >= 0.5 ? 'Malignant' : 'Benign'}`); // result for first sample - first result parameter (there is only 1 of each)
    console.log(`Test ${testYs === Math.round(result[0][0]) ? 'SUCCEEDED' : 'FAILED'}`);
    if (testYs === Math.round(result[0][0])) {
      success ++;
    }
  });
  console.log(`Times model evaluated to malignant: ${mals}`);
  console.log(`Times model evaluated to benign: ${bens}`);
  console.log(`Success rate: ${success}/${xs.slice(501).length}`);

  console.log(`Success rate during training: ${JSON.stringify(_.last(history.history.acc) * 100)}`);
}


runTrain();


// // Solve for XOR
// const LEARNING_RATE = 0.1;
// const EPOCHS = 200;

// // Define the training data
// const xs = [[0,0],[0,1],[1,0],[1,1]];
// const ys = [0,1,1,0];

// // Instantiate the training tensors
// let xTrain = tf.tensor2d(xs, [4,2]);
// let yTrain = tf.oneHot(tf.tensor1d(ys).toInt(), 2);

// // Define the model.
// const model = tf.sequential();
// // Set up the network layers
// model.add(tf.layers.dense({units: 5, activation: 'sigmoid', inputShape: [2]}));
// model.add(tf.layers.dense({units: 2, activation: 'softmax', outputShape: [2]}));
// // Define the optimizer
// const optimizer = tf.train.adam(LEARNING_RATE);
// // Init the model
// model.compile({
//     optimizer: optimizer,
//     loss: 'categoricalCrossentropy',
//     metrics: ['accuracy'],
// });
// // Train the model
// const history = model.fit(xTrain, yTrain, {
//   epochs: EPOCHS,
//   validationData: [xTrain, yTrain],
// }).then(()=>{
//   // Try the model on a value
//    const input = tf.tensor2d([0,1], [1, 2]);
//    const predictOut = model.predict(input);
//    const logits = Array.from(predictOut.dataSync());
//    console.log('prediction', logits, predictOut.argMax(-1).dataSync()[0]);
// });