class StaggerSwitch:
    def __init__(self, pde_names=["ac", "ch"], current_idx=-1):
        self.pde_names = pde_names
        self.current_idx = current_idx
        
    def switch(self, epoch: int, STAGGER_PERIOD: int):
        if epoch % STAGGER_PERIOD == 0:
            self.current_idx += 1
            self.current_idx %= len(self.pde_names)
            print(f"Switch to {self.pde_names[self.current_idx]} at epoch {epoch}")
        return self.pde_names[self.current_idx]