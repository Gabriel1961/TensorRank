import { Component } from "solid-js"
import styles from "./Navbar.module.scss"
import tfIcon from "../assets/tf.svg"
import ProfileImage from "./ProfileImage"
import ProfileDropdown from "./ProfileDropdown"
import { A } from "@solidjs/router"

const Navbar: Component = () => {
  return <div class={styles.container}>
    <A href="./" class={styles.logoContainer}>
      <img draggable={false} src={tfIcon} class={styles.logo} />
    </A>
    <A href="/" class={styles.header}>Home</A>
    <A href="/scoreboard" class={styles.header} activeClass={styles.actsiveHeader}>Scoreboard</A>
    <A href="/tasks" class={styles.header} activeClass={styles.activeHeader}>Tasks</A>
    <A href="/datasets" class={styles.header} activeClass={styles.activeHeader}>Datasets</A>
    <A href="/about" class={styles.header} activeClass={styles.activeHeader}>About</A>

    <div class={styles.profileImageContainer}>
      <ProfileDropdown profileImage={ProfileImage} />
    </div>
  </div>
}

export default Navbar;