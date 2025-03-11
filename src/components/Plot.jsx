import React from 'react';
import Plot from 'react-plotly.js';

const NormalDistributionPlot = ({ gradeDistribution }) => {
  console.log("HELLOOOO", gradeDistribution)
  // Prepare the data for the histogram
  const data = [{
      x: gradeDistribution,
      type: 'histogram',
  }];

  // Define layout options
  const layout = {
      title: 'Soft Grade',
      xaxis: {title: 'Score'},
      yaxis: {title: 'Likelihood'},
      bargap: 0.2 // Adjust gap between bars (optional)
  };

  return (
    <>
      <Plot data={data} layout={layout} />
    </>
  );
};

export default NormalDistributionPlot;