import React from 'react';
import { Button } from 'react-bootstrap';

const TeacherContainer = () => {
  const handleDownload = () => {
    const link = document.createElement('a');
    link.href = 'teacher_scripts.zip'; // Update this path to the actual file location
    link.download = 'soft_grades_teacher_scripts.zip';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div style={{ maxWidth: '900px', margin: '0 auto', padding: '40px', fontFamily: 'Arial, sans-serif', lineHeight: '1.6' }}>
      <h1 style={{ textAlign: 'center', borderBottom: '2px solid #ddd', paddingBottom: '10px' }}><b>Soft Grades:</b> A Tool for Teachers</h1>
      
      <h2>What are Soft Grades?</h2>
      <p>
        Soft Grades provide a way to capture the inherent uncertainty in academic performance. Traditional grading systems often assume that a student's score on an assessment is a perfect measure of their ability. However, performance can fluctuate due to factors unrelated to true understanding, such as sleep deprivation, personal circumstances, or inconsistencies in grading—whether from subjective rubrics or variability in how different graders, such as novice TAs, apply scoring criteria. Soft Grades models this uncertainties in a principled way by representing a student's performance as a probability distribution rather than a fixed score. This approach can help educators make more informed decisions about student progress and support.
      </p>
      
      <p>
      The Soft Grades approach is built on Course-Grade Response Theory, an extension of Item Response Theory designed to predict a student's final course grade while accounting for uncertainty. This method was developed by <strong>Juliette Woodrow</strong> and <strong>Chris Piech</strong>.  You can learn more about it in our research paper: <a href="https://juliettewoodrow.github.io/pdfs/SoftGrades.pdf" target="_blank">
        Soft Grades: A Calibrated and Accurate Method for Course-Grade Estimation that Expresses Uncertainty
      </a>.
      </p>
      
      <h2>1. Download the Scripts</h2>
      <p>
        Click the button below to download the necessary Python scripts. The downloaded directory contains:
      </p>
      <ul>
        <li><code>create_soft_grades.py</code> - Generates soft grades from assessment data.</li>
        <li><code>view_soft_grade.py</code> - Visualizes a student's soft grade.</li>
        <li><code>investigate_sg_stddev.py</code> - Analyzes and plots the distribution of soft grade standard deviations.</li>
        <li><code>requirements.txt</code> - Lists required Python dependencies.</li>
      </ul>
      <Button onClick={handleDownload} style={{ padding: '10px 20px', fontSize: '16px' }}>Download Python Scripts</Button>
      
      <div style={{ margin: '40px 0' }}></div>
      
      <h2>2. Install Dependencies</h2>
      <p>Ensure you have Python installed. Then, navigate to the downloaded directory and install the required libraries:</p>
      <pre><code>pip install -r requirements.txt</code></pre>

      <div style={{ margin: '40px 0' }}></div>
      
      <h2>3. Prepare Your CSV File</h2>
      <p>Your CSV file should contain a row for each student and columns for their scores on assessments. Scores must be numbers between 0 and 1 (e.g., 0.7 for 70%).</p>
      <table border="1" cellPadding="5" cellSpacing="0" style={{"border-collapse": "collapse", "width": "100%", "text-align": "left"}}>
        <thead>
          <tr>
            <th>StudentID</th>
            <th>Assessment1</th>
            <th>Assessment2</th>
            <th>Assessment3</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>S001</td>
            <td>0.8</td>
            <td>0.9</td>
            <td>0.75</td>
          </tr>
          <tr>
            <td>S002</td>
            <td>0.7</td>
            <td>0.85</td>
            <td>0.65</td>
          </tr>
          <tr>
            <td>S003</td>
            <td>0.9</td>
            <td>0.95</td>
            <td>0.8</td>
          </tr>
        </tbody>
      </table>

      <div style={{ margin: '40px 0' }}></div>
      
      <h2>4. Compute Soft Grades</h2>
      <p>Run the following command, providing the CSV file path:</p>
      <pre><code>python create_soft_grades.py path/to/assessment_scores.csv</code></pre>
      <p>The script will prompt you for the assessment columns and student ID column. It will then generate <code>soft_grades.json</code>.</p>
      
      <h2>5. View Soft Grades</h2>
      <p>Set the <code>STUDENT_ID</code> in <code>view_soft_grade.py</code> and run:</p>
      <pre><code>python view_soft_grade.py</code></pre>
      
      <h2>6. Investigate Standard Deviations</h2>
      <p>To analyze the standard deviation of students' soft grades, run:</p>
      <pre><code>python investigate_sg_stddev.py</code></pre>
      <p>This will generate <code>student_id_to_stddev.json</code> and a histogram of standard deviations.</p>
      
      <h2>7. Interpreting Results</h2>
      <ul>
        <li><strong>Soft grade distribution:</strong> Represents the uncertainty in an individual student's performance. A wider distribution (higher standard deviation) indicates greater variability in the student's grades, which could be useful for determining final grades—especially for students on a grade boundary (e.g., C+/B-). This can help teachers assess whether a student's performance is consistent or if there is significant fluctuation.</li>
        
        <li><strong>Histogram of standard deviations:</strong> Provides insight into the overall confidence in grades across an entire course. If many students have high standard deviations, it could mean greater uncertainty in grading or lots of uncertainty in students, whereas lower standard deviations indicate more consistent grading and student performance. This can be useful for comparing different course offerings over time to see whether this consistency has changed and to analyze patterns in student performance across different cohorts.</li>
      </ul>

      
      <h2>More Information</h2>
      <p>For details, read our research paper: <a href="https://juliettewoodrow.github.io/pdfs/SoftGrades.pdf" target="_blank">Soft Grades: A Calibrated and Accurate Method for Course-Grade
      Estimation that Expresses Uncertainty</a></p>
      <p>For questions, contact me at <a href="mailto:jwoodrow@stanford.edu">jwoodrow@stanford.edu</a>.</p>
    </div>
  );
};

export default TeacherContainer;
