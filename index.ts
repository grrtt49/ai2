import * as fs from 'fs';
import * as Papa from 'papaparse';
import { IsolationForest } from 'isolation-forest';

// Step 1: Read the CSV file
const csvData: any[] = [];
fs.createReadStream('claims_data.csv')
  .pipe(Papa.parse(Papa.NODE_STREAM_INPUT, {
    header: true,
    dynamicTyping: true,
  }))
  .on('data', (row) => {
    csvData.push(row);
  })
  .on('end', () => {
    // Step 2: Extract and preprocess the data
    const data = csvData.map((row) => ({
      DOB: new Date(row.DOB),
      SEX: row.SEX,
      ID: row.ID,
      DOS: new Date(row.DOS),
      CLAIM: row.CLAIM,
    }));

    // Step 3: Feature selection and transformation (if needed)
    // You may want to encode categorical variables like 'SEX' and 'CLAIM'.

    // Step 4: Train the Isolation Forest model
    const features = ['DOB', 'DOS'];
    const model = new IsolationForest();
    model.fit(data.map((row) => [{time: row.DOB.getTime()}, row.DOS.getTime()]));

    // Step 5: Make predictions on the same dataset
    const predictions = model.predict(data.map((row) => [row.DOB.getTime(), row.DOS.getTime()]));

    // Step 6: Detect anomalies
    const anomalyIndices = predictions.map((prediction, index) => (prediction === -1 ? index : -1)).filter((index) => index !== -1);

    // Step 7: Print or process the anomalies
    console.log('Anomalies:');
    for (const index of anomalyIndices) {
      console.log('ID:', data[index].ID);
      console.log('DOB:', data[index].DOB);
      console.log('SEX:', data[index].SEX);
      console.log('DOS:', data[index].DOS);
      console.log('CLAIM:', data[index].CLAIM);
      console.log('----------------');
    }
  });
