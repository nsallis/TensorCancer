const transformRow =
  (data) => {
    const values = [
      parseInt(data[2] * 10, 10), // radius_mean
      parseInt(data[3] * 10, 10), // texture mean
      parseInt(data[4] * 10, 10), // perimeter mean
      parseInt(data[5] / 10, 10), // area mean
      parseInt(data[6] * 100, 10), // smoothness mean
      parseInt(data[7] * 100, 10), // compactness mean
      parseInt(data[8] * 100, 10), // concavity mean
      parseInt(data[9] * 100, 10), // concavity points mean
      parseInt(data[10] * 100, 10), // symmetry mean
      parseInt(data[11] * 100, 10), // fractal dimension mean        
    ];
    return { xs: values, ys: (data[1] === 'M' ? 1 : 0) }; // 1 = malignant / 0 = benign
  }

function getRandomInt(max) {
  return Math.floor(Math.random() * Math.floor(max));
}

function shuffleArray(array) {
    for (let i = array.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [array[i], array[j]] = [array[j], array[i]];
    }
}

module.exports = {
  transformRow,
  getRandomInt,
  shuffleArray
}