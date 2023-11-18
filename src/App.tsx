import { Route, Router, Routes } from "@solidjs/router";
import { Component, lazy } from "solid-js";
import HomePage from "./pages/HomePage/HomePage";
import Navbar from "./components/Navbar";
import { LoginProvider } from "./contexts/loginContext";
import styles from "./components/Navbar.module.scss"


const AboutPage = lazy(() => import("./pages/AboutPage"))
const TasksPage = lazy(() => import("./pages/TasksPage/TasksPage"))
const TaskPage = lazy(() => import("./pages/TaskPage/TaskPage"))
const ScoreboardPage = lazy(() => import("./pages/Scoreboard/ScoreboardPage"))
const DatasetsPage = lazy(() => import("./pages/Datasets/DatasetsPage"))

export const App: Component = () => {
  return <LoginProvider>
    <Router>
      <Navbar />
      <div class={styles.phantomNav}></div>
      <Routes>
        <Route path="/" component={HomePage} />
        <Route path="/task/:taskName" component={TaskPage} />
        <Route path="/tasks" component={TasksPage} />
        <Route path="/scoreboard" component={ScoreboardPage} />
        <Route path="/datasets" component={DatasetsPage} />
        <Route path="/about" component={AboutPage} />
      </Routes>
    </Router >
  </LoginProvider>
}