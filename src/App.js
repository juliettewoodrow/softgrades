// App.js
import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import HomePage from './components/HomePage';
import AssignmentContainer from './components/AssignmentContainer';
import TeacherContainer from './components/TeacherContainer'; 
// <-- Create this if you haven’t already.

const App = () => {
  return (
    <Router>
      <Routes>
        {/* Home ("/") */}
        <Route path="/" element={<HomePage />} />

        {/* Student ("/student") */}
        <Route path="/student" element={<AssignmentContainer />} />

        {/* Teacher ("/teacher") */}
        <Route path="/teacher" element={<TeacherContainer />} />
      </Routes>
    </Router>
  );
};

export default App;
