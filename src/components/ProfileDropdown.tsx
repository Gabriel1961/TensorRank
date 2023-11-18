import { DropdownMenu } from "@kobalte/core";
import { Accessor, Component, ParentComponent, Show, createSignal } from "solid-js";
import googleIcon from "../assets/google.svg"
import styles from "./ProfileDropdown.module.scss"
import { GoogleAuthProvider, signInWithPopup, signOut } from "firebase/auth";
import { auth } from "../firebaseConfig";
import { useLogin } from "../contexts/loginContext";
interface ProfileDropdownProps {
  profileImage: Component<{ onClick: () => void, getImageUrl: Accessor<string | undefined> }>,
}

const ProfileDropdown: ParentComponent<ProfileDropdownProps> = (props) => {
  const [getIsOpen, setIsOpen] = createSignal(false)
  const [getLoginContext] = useLogin();
  const onLoginClick = async () => {
    try {
      const provider = new GoogleAuthProvider()
      await signInWithPopup(auth, provider)
    } catch (ex) {
      console.log(ex)
    }
  }

  return <DropdownMenu.Root>
    <DropdownMenu.Trigger class={styles["dropdown-menu__trigger"]} >
      <props.profileImage onClick={() => setIsOpen(!getIsOpen())} getImageUrl={() => getLoginContext().user?.photoURL ?? undefined} />
    </DropdownMenu.Trigger>
    <DropdownMenu.Portal>
      <DropdownMenu.Content class={styles["dropdown-menu__content"]}>
        <Show when={!getLoginContext().user && !getLoginContext().loading}>
          <DropdownMenu.Item class={styles["dropdown-menu__item"]} onClick={onLoginClick}>
            <div class={styles.loginContainer}>
              <div>Login</div>
              <img draggable={false} src={googleIcon} class={styles.googleImg} />
            </div>
          </DropdownMenu.Item>
        </Show>
        <Show when={getLoginContext().user && !getLoginContext().loading}>
          <DropdownMenu.Item class={styles["dropdown-menu__item"]} onClick={() => signOut(auth)}>
            <div>
              Log out
            </div>
          </DropdownMenu.Item>
        </Show>
      </DropdownMenu.Content>
    </DropdownMenu.Portal>
  </DropdownMenu.Root>
}
export default ProfileDropdown;